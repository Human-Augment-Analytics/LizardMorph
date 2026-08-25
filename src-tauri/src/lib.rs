use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use tauri::{webview::PageLoadEvent, Manager};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

const BACKEND_PORT: u16 = 3005;
const BACKEND_READY_TIMEOUT: Duration = Duration::from_secs(180);
const BACKEND_UNSUPERVISED_READY_TIMEOUT: Duration = Duration::from_secs(60);
const BACKEND_RESPAWN_DELAY: Duration = Duration::from_millis(1500);
const BACKEND_SPAWN_RETRIES: u32 = 2;
const PROBE_CONNECT_TIMEOUT: Duration = Duration::from_millis(150);
const PROBE_IO_TIMEOUT: Duration = Duration::from_millis(1000);
const PROBE_TOTAL_TIMEOUT: Duration = Duration::from_millis(2000);
const PROBE_INTERVAL: Duration = Duration::from_millis(250);
const PROBE_RESPONSE_LIMIT: usize = 4096;
const ORIGIN_PROBE_ATTEMPTS: u32 = 3;

const BACKEND_PORT_IN_USE_MESSAGE: &str = "AutoMorph could not start its analysis backend because another program is already using port 3005. Quit that program — including any earlier copy of AutoMorph — and open AutoMorph again.";
const BACKEND_START_FAILED_MESSAGE: &str =
    "AutoMorph could not start its analysis backend. Restart AutoMorph; if this keeps happening, reinstall it.";
const BACKEND_STOPPED_MESSAGE: &str =
    "AutoMorph's analysis backend stopped unexpectedly. Restart AutoMorph to continue.";
const BACKEND_NOT_RUNNING_MESSAGE: &str = "AutoMorph's analysis backend is not answering on 127.0.0.1:3005. Start it with 'make dev' (or 'make dev-backend') and reload this window.";
const BACKEND_ORIGIN_REJECTED_MESSAGE: &str = "AutoMorph's analysis backend refused this window's requests. Restart AutoMorph; if this keeps happening, reinstall it.";

#[cfg(windows)]
const DESKTOP_WEBVIEW_ORIGIN: &str = "http://tauri.localhost";
#[cfg(not(windows))]
const DESKTOP_WEBVIEW_ORIGIN: &str = "tauri://localhost";

#[derive(Default)]
struct BackendProcess {
    child: Mutex<Option<CommandChild>>,
    stopped: AtomicBool,
    ready: AtomicBool,
    shutting_down: AtomicBool,
    failure: Mutex<Option<&'static str>>,
}

fn overlay_script(message: &str) -> String {
    let message = serde_json::to_string(message)
        .unwrap_or_else(|_| String::from("\"AutoMorph's analysis backend is unavailable.\""));
    format!(
        "(function(){{var m={message};function render(){{var e=document.getElementById('automorph-backend-error');if(!e){{e=document.createElement('div');e.id='automorph-backend-error';e.setAttribute('role','alert');e.style.cssText='position:fixed;top:0;left:0;right:0;bottom:0;z-index:2147483647;display:flex;align-items:center;justify-content:center;padding:32px;background:#1b1b1f;color:#f5f5f5;font:16px/1.5 system-ui,sans-serif;text-align:center';(document.body||document.documentElement).appendChild(e);}}e.textContent=m;}}if(document.readyState==='loading'){{document.addEventListener('DOMContentLoaded',render);}}else{{render();}}}})()"
    )
}

/// Performs one bounded `GET /health` exchange and returns the raw response.
fn probe_health_endpoint(address: &SocketAddr, origin: Option<&str>) -> Option<String> {
    let probe_deadline = Instant::now() + PROBE_TOTAL_TIMEOUT;

    let mut stream = TcpStream::connect_timeout(address, PROBE_CONNECT_TIMEOUT).ok()?;
    if stream.set_read_timeout(Some(PROBE_IO_TIMEOUT)).is_err()
        || stream.set_write_timeout(Some(PROBE_IO_TIMEOUT)).is_err()
    {
        return None;
    }

    let origin_header = match origin {
        Some(origin) => format!("Origin: {origin}\r\n"),
        None => String::new(),
    };
    let request = format!(
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{BACKEND_PORT}\r\n{origin_header}Connection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return None;
    }

    let mut response = Vec::new();
    let mut buffer = [0u8; 512];
    while response.len() < PROBE_RESPONSE_LIMIT {
        if Instant::now() >= probe_deadline {
            return None;
        }
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => response.extend_from_slice(&buffer[..read]),
            Err(_) => break,
        }
    }

    Some(String::from_utf8_lossy(&response).into_owned())
}

fn split_http_response(response: &str) -> (&str, &str) {
    let mut sections = response.splitn(2, "\r\n\r\n");
    let head = sections.next().unwrap_or_default();
    let body = sections.next().unwrap_or_default();
    (head, body)
}

fn http_status_is_ok(head: &str) -> bool {
    let status_line = head.lines().next().unwrap_or_default();
    status_line.starts_with("HTTP/1.") && status_line.contains(" 200")
}

fn http_header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim().eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

/// Confirms the listener on `address` is this app's own backend: it must answer
/// `/health` with `status: ok` and echo `expected_supervisor` when one is given.
fn backend_is_healthy(address: &SocketAddr, expected_supervisor: Option<&str>) -> bool {
    let Some(response) = probe_health_endpoint(address, None) else {
        return false;
    };
    let (head, raw_body) = split_http_response(&response);
    if !http_status_is_ok(head) {
        return false;
    }

    let (Some(start), Some(end)) = (raw_body.find('{'), raw_body.rfind('}')) else {
        return false;
    };
    let Ok(payload) = serde_json::from_str::<serde_json::Value>(&raw_body[start..=end]) else {
        return false;
    };
    if payload.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
        return false;
    }

    match expected_supervisor {
        Some(expected) => {
            payload
                .get("supervisor_pid")
                .and_then(serde_json::Value::as_str)
                == Some(expected)
        }
        None => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OriginVerdict {
    Allowed,
    Rejected,
    Inconclusive,
}

/// A healthy backend that rejects this webview's `Origin` leaves the window
/// visible but unable to complete a single request, so it is checked explicitly.
fn probe_webview_origin(address: &SocketAddr) -> OriginVerdict {
    let Some(response) = probe_health_endpoint(address, Some(DESKTOP_WEBVIEW_ORIGIN)) else {
        return OriginVerdict::Inconclusive;
    };
    let (head, _) = split_http_response(&response);
    if !http_status_is_ok(head) {
        return OriginVerdict::Inconclusive;
    }
    match http_header(head, "access-control-allow-origin") {
        Some(allowed)
            if allowed == "*" || allowed.eq_ignore_ascii_case(DESKTOP_WEBVIEW_ORIGIN) =>
        {
            OriginVerdict::Allowed
        }
        _ => OriginVerdict::Rejected,
    }
}

fn webview_origin_verdict(address: &SocketAddr) -> OriginVerdict {
    for attempt in 0..ORIGIN_PROBE_ATTEMPTS {
        match probe_webview_origin(address) {
            OriginVerdict::Inconclusive => {
                if attempt + 1 < ORIGIN_PROBE_ATTEMPTS {
                    thread::sleep(PROBE_INTERVAL);
                }
            }
            verdict => return verdict,
        }
    }
    OriginVerdict::Inconclusive
}

/// A listener still holding the port after the sidecar gave up means the port is
/// taken by something else, not that the sidecar itself is broken.
fn backend_failure_message(address: &SocketAddr) -> &'static str {
    if TcpStream::connect_timeout(address, PROBE_CONNECT_TIMEOUT).is_ok() {
        BACKEND_PORT_IN_USE_MESSAGE
    } else {
        BACKEND_START_FAILED_MESSAGE
    }
}

/// Without a supervisor token this app did not spawn the backend, so the wait is
/// sized for a developer-run server and the copy points at it instead of the bundle.
fn readiness_timeout(supervisor: Option<&str>) -> Duration {
    if supervisor.is_some() {
        BACKEND_READY_TIMEOUT
    } else {
        BACKEND_UNSUPERVISED_READY_TIMEOUT
    }
}

fn unavailable_message(supervisor: Option<&str>, address: &SocketAddr) -> &'static str {
    if supervisor.is_none() {
        return BACKEND_NOT_RUNNING_MESSAGE;
    }
    backend_failure_message(address)
}

fn report_backend_unavailable(app_handle: &tauri::AppHandle, message: &'static str) {
    *app_handle
        .state::<BackendProcess>()
        .failure
        .lock()
        .unwrap() = Some(message);
    if let Some(window) = app_handle.get_webview_window("main") {
        if let Err(error) = window.eval(overlay_script(message)) {
            eprintln!("Failed to report the AutoMorph backend failure: {error}");
        }
    }
}

fn supervise_backend(app_handle: tauri::AppHandle, supervisor: Option<String>) {
    tauri::async_runtime::spawn_blocking(move || {
        let address = SocketAddr::from(([127, 0, 0, 1], BACKEND_PORT));
        let ready_timeout = readiness_timeout(supervisor.as_deref());
        let deadline = Instant::now() + ready_timeout;
        let mut respawns_left = if supervisor.is_some() {
            BACKEND_SPAWN_RETRIES
        } else {
            0
        };

        let mut ready = false;
        while Instant::now() < deadline {
            if backend_is_healthy(&address, supervisor.as_deref()) {
                ready = true;
                break;
            }
            thread::sleep(PROBE_INTERVAL);

            if !app_handle
                .state::<BackendProcess>()
                .stopped
                .load(Ordering::SeqCst)
            {
                continue;
            }

            let Some(token) = supervisor.as_deref() else {
                break;
            };
            if respawns_left == 0 {
                break;
            }
            respawns_left -= 1;
            thread::sleep(BACKEND_RESPAWN_DELAY);
            if Instant::now() >= deadline {
                break;
            }
            eprintln!("AutoMorph backend sidecar exited before becoming ready; retrying");
            if let Err(error) = spawn_backend(&app_handle, token) {
                eprintln!("Failed to restart the AutoMorph backend: {error}");
                break;
            }
        }

        if !ready {
            eprintln!(
                "AutoMorph backend never answered /health on 127.0.0.1:{BACKEND_PORT} within {} seconds",
                ready_timeout.as_secs()
            );
        }

        if ready
            && app_handle
                .state::<BackendProcess>()
                .stopped
                .load(Ordering::SeqCst)
        {
            eprintln!("AutoMorph backend answered /health but its sidecar has already exited");
            ready = false;
        }

        let mut origin_rejected = false;
        if ready {
            match webview_origin_verdict(&address) {
                OriginVerdict::Allowed => {}
                OriginVerdict::Rejected => {
                    eprintln!(
                        "AutoMorph backend is healthy but rejects the webview origin {DESKTOP_WEBVIEW_ORIGIN}"
                    );
                    origin_rejected = true;
                    ready = false;
                }
                OriginVerdict::Inconclusive => {
                    eprintln!(
                        "AutoMorph backend stopped answering before its origin policy could be checked"
                    );
                    ready = false;
                }
            }
        }

        if ready {
            app_handle
                .state::<BackendProcess>()
                .ready
                .store(true, Ordering::SeqCst);
        } else if origin_rejected {
            report_backend_unavailable(&app_handle, BACKEND_ORIGIN_REJECTED_MESSAGE);
        } else {
            report_backend_unavailable(
                &app_handle,
                unavailable_message(supervisor.as_deref(), &address),
            );
        }

        if let Some(window) = app_handle.get_webview_window("main") {
            if let Err(error) = window.show() {
                eprintln!("Failed to show the AutoMorph window: {error}");
            }
        }
    });
}

fn spawn_backend(
    app_handle: &tauri::AppHandle,
    supervisor: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    app_handle
        .state::<BackendProcess>()
        .stopped
        .store(false, Ordering::SeqCst);

    let command = app_handle
        .shell()
        .sidecar("python-backend")?
        .env("API_PORT", BACKEND_PORT.to_string())
        .env("AUTOMORPH_PARENT_PID", supervisor)
        .env("PYTHONUNBUFFERED", "1");
    let (mut events, child) = command.spawn()?;
    let pid = child.pid();
    *app_handle
        .state::<BackendProcess>()
        .child
        .lock()
        .unwrap() = Some(child);

    let app_handle = app_handle.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    println!("[backend] {}", String::from_utf8_lossy(&bytes))
                }
                CommandEvent::Stderr(bytes) => {
                    eprintln!("[backend] {}", String::from_utf8_lossy(&bytes))
                }
                CommandEvent::Error(error) => eprintln!("[backend] {error}"),
                CommandEvent::Terminated(payload) => {
                    eprintln!(
                        "AutoMorph backend process {pid} exited (code: {:?}, signal: {:?})",
                        payload.code, payload.signal
                    );
                    let lost_while_running = {
                        let state = app_handle.state::<BackendProcess>();
                        state.stopped.store(true, Ordering::SeqCst);
                        state.child.lock().unwrap().take();
                        state.ready.load(Ordering::SeqCst)
                            && !state.shutting_down.load(Ordering::SeqCst)
                    };
                    if lost_while_running {
                        report_backend_unavailable(&app_handle, BACKEND_STOPPED_MESSAGE);
                    }
                    break;
                }
                _ => {}
            }
        }
    });
    Ok(())
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(BackendProcess::default())
        .on_page_load(|webview, payload| {
            if payload.event() != PageLoadEvent::Finished {
                return;
            }
            let failure = *webview.state::<BackendProcess>().failure.lock().unwrap();
            if let Some(message) = failure {
                if let Err(error) = webview.eval(overlay_script(message)) {
                    eprintln!("Failed to report the AutoMorph backend failure: {error}");
                }
            }
        })
        .setup(|app| {
            let supervisor = if tauri::is_dev() {
                None
            } else {
                let supervisor = std::process::id().to_string();
                if let Err(error) = spawn_backend(app.handle(), &supervisor) {
                    eprintln!("Failed to start the AutoMorph backend: {error}");
                    app.state::<BackendProcess>()
                        .stopped
                        .store(true, Ordering::SeqCst);
                }
                Some(supervisor)
            };
            supervise_backend(app.handle().clone(), supervisor);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building AutoMorph");

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            let state = app_handle.state::<BackendProcess>();
            state.shutting_down.store(true, Ordering::SeqCst);
            let child = state.child.lock().unwrap().take();
            if let Some(child) = child {
                if let Err(error) = child.kill() {
                    eprintln!("Failed to stop AutoMorph backend: {error}");
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    fn json_response(body: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    fn cors_response(allow_origin: Option<&str>) -> Vec<u8> {
        let body = r#"{"status":"ok"}"#;
        let allow_header = match allow_origin {
            Some(origin) => format!("Access-Control-Allow-Origin: {origin}\r\n"),
            None => String::new(),
        };
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{allow_header}Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    fn read_request(stream: &TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            if line == "\r\n" {
                break;
            }
            line.clear();
        }
    }

    fn spawn_listener(response: Option<Vec<u8>>) -> SocketAddr {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                read_request(&stream);
                if let Some(body) = response {
                    let _ = stream.write_all(&body);
                }
            }
        });
        address
    }

    fn closed_port() -> SocketAddr {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        address
    }

    fn serve_responses(mut responses: Vec<Option<Vec<u8>>>) -> SocketAddr {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        responses.reverse();
        thread::spawn(move || {
            while let Some(response) = responses.pop() {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                read_request(&stream);
                match response {
                    Some(body) => {
                        let _ = stream.write_all(&body);
                    }
                    None => drop(stream),
                }
            }
        });
        address
    }

    #[test]
    fn origin_probe_accepts_a_backend_that_echoes_this_webview_origin() {
        let address = spawn_listener(Some(cors_response(Some(DESKTOP_WEBVIEW_ORIGIN))));
        assert_eq!(probe_webview_origin(&address), OriginVerdict::Allowed);
    }

    #[test]
    fn origin_probe_accepts_a_backend_that_allows_every_origin() {
        let address = spawn_listener(Some(cors_response(Some("*"))));
        assert_eq!(probe_webview_origin(&address), OriginVerdict::Allowed);
    }

    #[test]
    fn origin_probe_rejects_a_backend_that_omits_the_allow_origin_header() {
        let address = spawn_listener(Some(cors_response(None)));
        assert_eq!(probe_webview_origin(&address), OriginVerdict::Rejected);
    }

    #[test]
    fn origin_probe_rejects_a_backend_that_allows_only_another_origin() {
        let address = spawn_listener(Some(cors_response(Some("https://example.test"))));
        assert_eq!(probe_webview_origin(&address), OriginVerdict::Rejected);
    }

    #[test]
    fn origin_probe_reports_a_dead_port_as_inconclusive_not_rejected() {
        assert_eq!(
            probe_webview_origin(&closed_port()),
            OriginVerdict::Inconclusive
        );
    }

    #[test]
    fn origin_probe_reports_a_dropped_connection_as_inconclusive_not_rejected() {
        let address = spawn_listener(None);
        assert_eq!(probe_webview_origin(&address), OriginVerdict::Inconclusive);
    }

    #[test]
    fn origin_probe_reports_a_non_200_response_as_inconclusive_not_rejected() {
        let address = spawn_listener(Some(
            b"HTTP/1.1 503 SERVICE UNAVAILABLE\r\nContent-Length: 0\r\n\r\n".to_vec(),
        ));
        assert_eq!(probe_webview_origin(&address), OriginVerdict::Inconclusive);
    }

    #[test]
    fn origin_verdict_retries_past_a_transient_transport_failure() {
        let address = serve_responses(vec![
            None,
            None,
            Some(cors_response(Some(DESKTOP_WEBVIEW_ORIGIN))),
        ]);
        assert_eq!(webview_origin_verdict(&address), OriginVerdict::Allowed);
    }

    #[test]
    fn origin_verdict_stays_inconclusive_when_every_attempt_fails() {
        assert_eq!(
            webview_origin_verdict(&closed_port()),
            OriginVerdict::Inconclusive
        );
    }

    #[test]
    fn origin_verdict_does_not_retry_a_completed_rejection() {
        let address = serve_responses(vec![
            Some(cors_response(Some("https://example.test"))),
            Some(cors_response(Some(DESKTOP_WEBVIEW_ORIGIN))),
        ]);
        assert_eq!(webview_origin_verdict(&address), OriginVerdict::Rejected);
    }

    #[test]
    fn origin_probe_sends_this_webview_origin() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                if line == "\r\n" {
                    break;
                }
                request.push_str(&line);
                line.clear();
            }
            let _ = stream.write_all(&cors_response(Some(DESKTOP_WEBVIEW_ORIGIN)));
            request
        });

        assert_eq!(probe_webview_origin(&address), OriginVerdict::Allowed);
        let request = handle.join().unwrap();
        assert!(request.contains(&format!("Origin: {DESKTOP_WEBVIEW_ORIGIN}\r\n")));
    }

    #[test]
    fn health_probe_sends_no_origin_header() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                if line == "\r\n" {
                    break;
                }
                request.push_str(&line);
                line.clear();
            }
            let _ = stream.write_all(&json_response(r#"{"status":"ok"}"#));
            request
        });

        assert!(backend_is_healthy(&address, None));
        let request = handle.join().unwrap();
        assert!(!request.to_ascii_lowercase().contains("origin:"));
    }

    #[test]
    fn health_probe_accepts_a_backend_health_response() {
        let address = spawn_listener(Some(json_response(r#"{"status":"ok"}"#)));
        assert!(backend_is_healthy(&address, None));
    }

    #[test]
    fn health_probe_rejects_a_listener_that_does_not_speak_http() {
        let address = spawn_listener(None);
        assert!(!backend_is_healthy(&address, None));
    }

    #[test]
    fn health_probe_rejects_a_non_200_response() {
        let address = spawn_listener(Some(
            b"HTTP/1.1 404 NOT FOUND\r\nContent-Length: 0\r\n\r\n".to_vec(),
        ));
        assert!(!backend_is_healthy(&address, None));
    }

    #[test]
    fn health_probe_rejects_a_port_with_no_listener() {
        assert!(!backend_is_healthy(&closed_port(), None));
    }

    #[test]
    fn health_probe_accepts_a_backend_supervised_by_this_process() {
        let address = spawn_listener(Some(json_response(
            r#"{"status":"ok","supervisor_pid":"4242"}"#,
        )));
        assert!(backend_is_healthy(&address, Some("4242")));
    }

    #[test]
    fn health_probe_rejects_a_backend_supervised_by_another_process() {
        let address = spawn_listener(Some(json_response(
            r#"{"status":"ok","supervisor_pid":"1111"}"#,
        )));
        assert!(!backend_is_healthy(&address, Some("4242")));
    }

    #[test]
    fn health_probe_rejects_an_unsupervised_backend_when_a_sidecar_is_expected() {
        let address = spawn_listener(Some(json_response(r#"{"status":"ok"}"#)));
        assert!(!backend_is_healthy(&address, Some("4242")));
    }

    #[test]
    fn health_probe_gives_up_on_a_listener_that_never_finishes_its_response() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                read_request(&stream);
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\n\r\n{");
                loop {
                    if stream.write_all(b" ").is_err() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            }
        });

        let started = Instant::now();
        assert!(!backend_is_healthy(&address, None));
        assert!(started.elapsed() < PROBE_TOTAL_TIMEOUT * 3);
    }

    #[test]
    fn failure_message_blames_the_port_when_something_still_holds_it() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        assert_eq!(backend_failure_message(&address), BACKEND_PORT_IN_USE_MESSAGE);
        drop(listener);
    }

    #[test]
    fn failure_message_blames_the_sidecar_when_the_port_is_free() {
        assert_eq!(
            backend_failure_message(&closed_port()),
            BACKEND_START_FAILED_MESSAGE
        );
    }

    #[test]
    fn an_unsupervised_backend_is_never_blamed_on_the_bundle() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let occupied = listener.local_addr().unwrap();

        assert_eq!(
            unavailable_message(None, &occupied),
            BACKEND_NOT_RUNNING_MESSAGE
        );
        assert_eq!(
            unavailable_message(None, &closed_port()),
            BACKEND_NOT_RUNNING_MESSAGE
        );
        assert_eq!(
            unavailable_message(Some("4242"), &occupied),
            BACKEND_PORT_IN_USE_MESSAGE
        );
        assert_eq!(
            unavailable_message(Some("4242"), &closed_port()),
            BACKEND_START_FAILED_MESSAGE
        );
        drop(listener);
    }

    #[test]
    fn an_unsupervised_backend_is_not_waited_on_for_the_packaged_timeout() {
        assert_eq!(readiness_timeout(Some("4242")), BACKEND_READY_TIMEOUT);
        assert!(readiness_timeout(None) < BACKEND_READY_TIMEOUT);
    }

    #[test]
    fn overlay_script_carries_the_message_through_json_escaping() {
        let message = r#"Port "3005" is busy \ retry"#;
        let script = overlay_script(message);

        let start = script.find("var m=").unwrap() + "var m=".len();
        let end = script[start..].find(";function render()").unwrap() + start;
        let embedded: String = serde_json::from_str(&script[start..end]).unwrap();

        assert_eq!(embedded, message);
    }
}
