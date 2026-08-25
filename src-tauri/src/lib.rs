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

use tauri::Manager;
use tauri_plugin_shell::process::CommandChild;
#[cfg(not(debug_assertions))]
use tauri_plugin_shell::{process::CommandEvent, ShellExt};

const BACKEND_PORT: u16 = 3005;
const BACKEND_READY_TIMEOUT: Duration = Duration::from_secs(180);
const PROBE_CONNECT_TIMEOUT: Duration = Duration::from_millis(150);
const PROBE_IO_TIMEOUT: Duration = Duration::from_millis(1000);
const PROBE_INTERVAL: Duration = Duration::from_millis(250);
const WEBVIEW_LOAD_GRACE: Duration = Duration::from_secs(2);

const BACKEND_FAILURE_OVERLAY: &str = r#"(function(){var m='AutoMorph could not start its analysis backend. Restart AutoMorph; if this keeps happening, reinstall it.';function render(){if(document.getElementById('automorph-backend-error'))return;var e=document.createElement('div');e.id='automorph-backend-error';e.setAttribute('role','alert');e.style.cssText='position:fixed;top:0;left:0;right:0;bottom:0;z-index:2147483647;display:flex;align-items:center;justify-content:center;padding:32px;background:#1b1b1f;color:#f5f5f5;font:16px/1.5 system-ui,sans-serif;text-align:center';e.textContent=m;(document.body||document.documentElement).appendChild(e);}if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',render);}else{render();}})()"#;

#[derive(Default)]
struct BackendProcess {
    child: Mutex<Option<CommandChild>>,
    stopped: AtomicBool,
}

/// Confirms the listener on `address` is an AutoMorph backend rather than an
/// unrelated process that happens to hold the port.
fn backend_is_healthy(address: &SocketAddr) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(address, PROBE_CONNECT_TIMEOUT) else {
        return false;
    };
    if stream.set_read_timeout(Some(PROBE_IO_TIMEOUT)).is_err()
        || stream.set_write_timeout(Some(PROBE_IO_TIMEOUT)).is_err()
    {
        return false;
    }

    let request =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{BACKEND_PORT}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }

    let mut response = Vec::new();
    let mut buffer = [0u8; 512];
    while response.len() < 4096 {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => response.extend_from_slice(&buffer[..read]),
            Err(_) => break,
        }
    }

    let response = String::from_utf8_lossy(&response);
    let mut sections = response.splitn(2, "\r\n\r\n");
    let status_line = sections.next().unwrap_or_default().lines().next().unwrap_or_default();
    let body = sections.next().unwrap_or_default();

    status_line.starts_with("HTTP/1.") && status_line.contains(" 200") && body.contains("\"status\"")
}

fn show_window_when_backend_is_ready(app_handle: tauri::AppHandle) {
    tauri::async_runtime::spawn_blocking(move || {
        let address = SocketAddr::from(([127, 0, 0, 1], BACKEND_PORT));
        let deadline = Instant::now() + BACKEND_READY_TIMEOUT;

        let mut ready = false;
        while Instant::now() < deadline {
            if backend_is_healthy(&address) {
                ready = true;
                break;
            }
            thread::sleep(PROBE_INTERVAL);
            if app_handle
                .state::<BackendProcess>()
                .stopped
                .load(Ordering::SeqCst)
            {
                break;
            }
        }

        if !ready {
            eprintln!(
                "AutoMorph backend never answered /health on 127.0.0.1:{BACKEND_PORT} within {} seconds",
                BACKEND_READY_TIMEOUT.as_secs()
            );
            thread::sleep(WEBVIEW_LOAD_GRACE);
        }

        if let Some(window) = app_handle.get_webview_window("main") {
            if let Err(error) = window.show() {
                eprintln!("Failed to show the AutoMorph window: {error}");
            }
            if !ready {
                if let Err(error) = window.eval(BACKEND_FAILURE_OVERLAY) {
                    eprintln!("Failed to report the AutoMorph backend failure: {error}");
                }
            }
        }
    });
}

#[cfg(not(debug_assertions))]
fn start_backend(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let command = app
        .shell()
        .sidecar("python-backend")?
        .env("API_PORT", BACKEND_PORT.to_string())
        .env("AUTOMORPH_PARENT_PID", std::process::id().to_string())
        .env("PYTHONUNBUFFERED", "1");
    let (mut events, child) = command.spawn()?;
    let pid = child.pid();
    *app.state::<BackendProcess>().child.lock().unwrap() = Some(child);
    let app_handle = app.handle().clone();

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
                    let state = app_handle.state::<BackendProcess>();
                    state.stopped.store(true, Ordering::SeqCst);
                    state.child.lock().unwrap().take();
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
        .setup(|app| {
            #[cfg(not(debug_assertions))]
            if let Err(error) = start_backend(app) {
                eprintln!("Failed to start the AutoMorph backend: {error}");
                app.state::<BackendProcess>()
                    .stopped
                    .store(true, Ordering::SeqCst);
            }
            show_window_when_backend_is_ready(app.handle().clone());
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building AutoMorph");

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            if let Some(child) = app_handle
                .state::<BackendProcess>()
                .child
                .lock()
                .unwrap()
                .take()
            {
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

    fn spawn_listener(response: Option<&'static [u8]>) -> SocketAddr {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if line == "\r\n" {
                        break;
                    }
                    line.clear();
                }
                if let Some(body) = response {
                    let _ = stream.write_all(body);
                }
            }
        });
        address
    }

    #[test]
    fn health_probe_accepts_a_backend_health_response() {
        let address = spawn_listener(Some(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 16\r\n\r\n{\"status\":\"ok\"}\n",
        ));
        assert!(backend_is_healthy(&address));
    }

    #[test]
    fn health_probe_rejects_a_listener_that_does_not_speak_http() {
        let address = spawn_listener(None);
        assert!(!backend_is_healthy(&address));
    }

    #[test]
    fn health_probe_rejects_a_non_200_response() {
        let address = spawn_listener(Some(b"HTTP/1.1 404 NOT FOUND\r\nContent-Length: 0\r\n\r\n"));
        assert!(!backend_is_healthy(&address));
    }

    #[test]
    fn health_probe_rejects_a_port_with_no_listener() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        assert!(!backend_is_healthy(&address));
    }
}
