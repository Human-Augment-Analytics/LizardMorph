use tauri_plugin_shell::ShellExt;

mod commands {
    use tauri_plugin_shell::ShellExt;

    #[tauri::command]
    pub async fn start_python_sidecar(app_handle: tauri::AppHandle) -> Result<String, String> {
        match app_handle.shell().sidecar("python-backend") {
            Ok(command) => match command.spawn() {
                Ok((_rx, child)) => Ok(format!("Python sidecar process spawned successfully (PID: {})", child.pid())),
                Err(e) => Err(format!("Failed to spawn python-backend sidecar: {}", e)),
            },
            Err(e) => Err(format!("Failed to configure python-backend sidecar: {}", e)),
        }
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                match handle.shell().sidecar("python-backend") {
                    Ok(command) => match command.spawn() {
                        Ok((_rx, child)) => {
                            println!("Auto-launched python-backend sidecar (PID: {})", child.pid());
                        }
                        Err(e) => {
                            eprintln!("Python sidecar binary not running yet (expected if binary not built yet): {}", e);
                        }
                    },
                    Err(e) => {
                        eprintln!("Could not configure sidecar command: {}", e);
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![commands::start_python_sidecar])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
