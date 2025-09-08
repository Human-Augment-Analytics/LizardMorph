import { fileURLToPath, URL } from 'node:url';
import { readFileSync } from 'fs';
import { defineConfig, loadEnv } from 'vite';
import plugin from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd() + "/../", '');

    const apiPort = env.VITE_API_PORT || env.API_PORT || '5000';
    const baseURL = env.VITE_BASE_URL || env.BASE_URL || '/';
    
    // Generate version for cache busting
    const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));
    const version = packageJson.version;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const buildVersion = `${version}-${timestamp}`;
    
    return {
        base: baseURL, 
        plugins: [plugin()],
        resolve: {
            alias: {
                '@': fileURLToPath(new URL('./src', import.meta.url))
            }
        },
        build: {
            // Add version to build output
            rollupOptions: {
                output: {
                    // Add version to chunk names for cache busting
                    chunkFileNames: `assets/[name]-${buildVersion}-[hash].js`,
                    entryFileNames: `assets/[name]-${buildVersion}-[hash].js`,
                    assetFileNames: `assets/[name]-${buildVersion}-[hash].[ext]`
                }
            },
            // Generate manifest for version tracking
            manifest: true,
            // Add sourcemap for debugging
            sourcemap: mode === 'development'
        },
        define: {
            // Make version available in the app
            __APP_VERSION__: JSON.stringify(version),
            __BUILD_VERSION__: JSON.stringify(buildVersion)
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
        }
    }
});

