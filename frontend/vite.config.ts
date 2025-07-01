import { fileURLToPath, URL } from 'node:url';

import { defineConfig, loadEnv } from 'vite';
import plugin from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd() + "/../", '');

    const apiPort = env.VITE_API_PORT || env.API_PORT || '5000';
    const baseURL = env.VITE_BASE_URL || env.BASE_URL || '/';
    return {
        base: baseURL, 
        plugins: [plugin()],
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
                }
            },
            allowedHosts: ['localhost', '127.0.0.1', '0.0.0.0', env.VITE_ALLOWED_HOSTS],
        }
    }
});

