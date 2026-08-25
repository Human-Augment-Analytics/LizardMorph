use std::{
    net::{SocketAddr, TcpStream},
    sync::Mutex,
    thread,
    time::Duration,
};

use tauri::Manager;
use tauri_plugin_shell::process::CommandChild;
#[cfg(not(debug_assertions))]
use tauri_plugin_shell::{process::CommandEvent, ShellExt};

#[derive(Default)]
struct BackendProcess(Mutex<Option<CommandChild>>);

fn show_window_when_backend_is_ready(app_handle: tauri::AppHandle) {
    tauri::async_runtime::spawn_blocking(move || {
        let address = SocketAddr::from(([127, 0, 0, 1], 3005));
        let mut ready = false;
        for _ in 0..720 {
            if TcpStream::connect_timeout(&address, Duration::from_millis(150)).is_ok() {
                ready = true;
                break;
            }
            thread::sleep(Duration::from_millis(250));
        }

        if !ready {
            eprintln!(
                "AutoMorph backend did not become ready on 127.0.0.1:3005 within 180 seconds"
            );
        }
        if let Some(window) = app_handle.get_webview_window("main") {
            if let Err(error) = window.show() {
                eprintln!("Failed to show the AutoMorph window: {error}");
            }
        }
    });
}

#[cfg(not(debug_assertions))]
fn start_backend(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let command = app
        .shell()
        .sidecar("python-backend")?
        .env("API_PORT", "3005")
        .env("AUTOMORPH_PARENT_PID", std::process::id().to_string())
        .env("PYTHONUNBUFFERED", "1");
    let (mut events, child) = command.spawn()?;
    let pid = child.pid();
    *app.state::<BackendProcess>().0.lock().unwrap() = Some(child);
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
                    app_handle
                        .state::<BackendProcess>()
                        .0
                        .lock()
                        .unwrap()
                        .take();
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
            start_backend(app)?;
            show_window_when_backend_is_ready(app.handle().clone());
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building AutoMorph");

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            if let Some(child) = app_handle
                .state::<BackendProcess>()
                .0
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
