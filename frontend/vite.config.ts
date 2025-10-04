import { fileURLToPath, URL } from 'node:url';

import { defineConfig, loadEnv } from 'vite';
import plugin from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd() + "/../", '');

    const apiPort = env.VITE_API_PORT || env.API_PORT || '5000';
    const baseURL = env.VITE_BASE_URL || env.BASE_URL || '/';
    return {
        base: baseURL,
        plugins: [
            wasm(),
            topLevelAwait(),
            plugin(),
        ],
        resolve: {
            alias: {
                '@': fileURLToPath(new URL('./src', import.meta.url))
            }
        },
        server: {
            proxy: {
                '^/api': {
                    target: `http://localhost:${apiPort}`,
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ''),
                }
            },
            allowedHosts: ['localhost', '127.0.0.1', '0.0.0.0', env.VITE_ALLOWED_HOSTS],
            headers: {
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Embedder-Policy': 'require-corp',
            },
        },
        optimizeDeps: {
            exclude: ['onnxruntime-web'],
        },
        assetsInclude: ['**/*.wasm'],
        build: {
            rollupOptions: {
                external: (id) => {
                    // Don't bundle WASM files, serve them as static assets
                    return id.includes('.wasm');
                }
            }
        },
    }
});

