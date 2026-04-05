import { execSync } from 'node:child_process';
import { fileURLToPath, URL } from 'node:url';
import { readFileSync, createReadStream, existsSync } from 'fs';
import path from 'node:path';
import { defineConfig, loadEnv } from 'vite';
import type { Plugin } from 'vite';
import plugin from '@vitejs/plugin-react';

/**
 * Vite plugin: serve onnxruntime-web .mjs worker files from node_modules in dev.
 * Files in public/ can't be import()-ed by Vite's dev server, but ORT needs to
 * dynamically import these as module workers for multi-threaded WASM.
 */
function ortWasmDevPlugin(): Plugin {
    return {
        name: 'ort-wasm-dev',
        configureServer(server) {
            server.middlewares.use((req, res, next) => {
                if (req.url && /\/ort-wasm.*\.mjs(\?|$)/.test(req.url)) {
                    const basename = req.url.split('?')[0].split('/').pop()!;
                    const filePath = path.join(__dirname, 'node_modules/onnxruntime-web/dist', basename);
                    if (existsSync(filePath)) {
                        res.setHeader('Content-Type', 'application/javascript');
                        res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
                        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
                        createReadStream(filePath).pipe(res);
                        return;
                    }
                }
                next();
            });
        },
    };
}

const repoRoot = fileURLToPath(new URL('..', import.meta.url));

/** For UI only: v1.0.1-13-g54f5ec5-dirty → v1.0.1 (full label stays in filenames / __BUILD_VERSION__). */
function displayReleaseLabel(fullLabel: string): string {
    let s = fullLabel.replace(/-\d+-g[0-9a-f]+(-dirty)?$/i, '');
    s = s.replace(/-dirty$/i, '');
    return s;
}

/** Full release label (git describe, CI tag, etc.) — unnormalized for cache-bust asset names. */
function resolveReleaseVersion(packageSemver: string): string {
    const fromEnv = process.env.VITE_RELEASE_VERSION?.trim();
    if (fromEnv && /^v\d/.test(fromEnv)) return fromEnv;

    const refName = process.env.GITHUB_REF_NAME?.trim();
    if (refName && /^v\d/.test(refName)) return refName;

    try {
        const desc = execSync('git describe --tags --match "v*" --always --dirty', {
            cwd: repoRoot,
            encoding: 'utf-8',
            stdio: ['ignore', 'pipe', 'ignore'],
        }).trim();
        if (desc) return desc;
    } catch {
        /* no .git, shallow clone, etc. */
    }

    return `v${packageSemver}`;
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd() + "/../", '');

    const apiPort = env.VITE_API_PORT || env.API_PORT || '5000';
    const baseURL = env.VITE_BASE_URL || './';
    
    const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));
    const releaseVersion = resolveReleaseVersion(packageJson.version);
    const releaseVersionDisplay = displayReleaseLabel(releaseVersion);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const buildVersion = `${releaseVersion}-${timestamp}`;
    
    return {
        base: baseURL, 
        plugins: [ortWasmDevPlugin(), plugin()],
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
            // Short tag-style label for Header / UI only
            __APP_VERSION__: JSON.stringify(releaseVersionDisplay),
            // Full label + timestamp (e.g. for support, asset names)
            __BUILD_VERSION__: JSON.stringify(buildVersion)
        },
        server: {
            headers: {
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Embedder-Policy': 'require-corp',
            },
            proxy: {
                '/api': {
                    target: `http://localhost:${apiPort}`,
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ''),
                }
            },
            allowedHosts: ['localhost', '127.0.0.1', '0.0.0.0', env.VITE_ALLOWED_HOSTS],
        }
    }
});

