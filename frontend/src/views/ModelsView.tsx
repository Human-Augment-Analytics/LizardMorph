import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useTheme } from "../contexts/theme";
import { getTokens } from "../contexts/themeTokens";
import { ModelsApiService, type ModelVersionItem, type ImportedModel } from "../services/ModelsApiService";

export function ModelsView() {
  const { resolved } = useTheme();
  const t = getTokens(resolved);
  const [models, setModels] = useState<ModelVersionItem[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [installed, setInstalled] = useState<Record<string, ImportedModel>>({});
  useEffect(() => {
    ModelsApiService.listModels().then(({ models }) => setModels(models.filter(model => model.tags.bundle_format === "lizard-pipeline-v1")))
      .catch(error => setError(String(error))).finally(() => setLoading(false));
  }, []);
  async function install(model: ModelVersionItem) {
    const key = `${model.name}:${model.version}`;
    setBusy(key); setError(null);
    try {
      const response = await ModelsApiService.importBundle(model.name, model.version);
      setInstalled(previous => ({ ...previous, [key]: response.model }));
    } catch (error) { setError(String(error)); }
    finally { setBusy(null); }
  }
  return <main style={{ minHeight: "100vh", background: t.bg, color: t.text, padding: "clamp(16px, 4vw, 48px)" }}>
    <div style={{ maxWidth: 960, margin: "0 auto" }}>
      <Link to="/" style={{ color: t.textLink }}>← Model catalog</Link>
      <h1>Research models</h1>
      <p>Import a complete model version, then open it to analyze images. Each analysis uses that exact version.</p>
      <p style={{ color: t.textSecondary }}>Experimental models require accuracy review before scientific use. Importing does not change the default model.</p>
      {loading && <p>Loading registry…</p>}
      {error && <p role="alert">{error}</p>}
      {!loading && models.length === 0 && !error && <p>No pipeline bundles registered yet.</p>}
      <div style={{ display: "grid", gap: 16 }}>
        {models.map(model => {
          const key = `${model.name}:${model.version}`;
          const imported = installed[key];
          return <article key={key} style={{ background: t.surface, border: `1px solid ${t.borderLight}`, borderRadius: 12, padding: 24, overflowWrap: "anywhere" }}>
            <h2 style={{ marginTop: 0 }}>{model.name} · v{model.version}</h2>
            <p>{model.tags.evaluation_status || "Evaluation not recorded"}</p>
            {model.aliases.length > 0 && <p>Aliases: {model.aliases.join(", ")}</p>}
            {imported ? <>
              <p>Validated and imported. Version {imported.registry.version} is pinned.</p>
              <Link to={`/custom-model/${encodeURIComponent(imported.id)}`} style={{ color: t.textLink }}>Use this version →</Link>
            </> : <button disabled={busy !== null} onClick={() => install(model)} style={{ padding: "10px 16px", cursor: "pointer" }}>
              {busy === key ? "Validating and importing…" : "Import this version"}
            </button>}
          </article>;
        })}
      </div>
    </div>
  </main>;
}
