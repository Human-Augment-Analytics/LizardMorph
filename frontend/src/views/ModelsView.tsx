import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useTheme } from "../contexts/ThemeContext";
import { getModelsViewStyles } from "./ModelsView.style";
import {
  ModelsApiService,
  type HealthResponse,
  type ExperimentItem,
  type RunItem,
  type LatestModelResponse,
  type ModelVersionItem,
} from "../services/ModelsApiService";

async function downloadFromTag(tag: string, asset: "fp16" | "fp32" | "pt") {
  try {
    const resp = await ModelsApiService.getModelByTag(tag);
    const url = resp.assets[asset];
    if (url) window.open(url, "_blank");
  } catch (e) {
    alert(`Failed to get download URL: ${e instanceof Error ? e.message : e}`);
  }
}

export function ModelsView() {
  const { resolved } = useTheme();
  const styles = getModelsViewStyles(resolved);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [latest, setLatest] = useState<LatestModelResponse | null>(null);
  const [latestError, setLatestError] = useState<string | null>(null);
  const [modelVersions, setModelVersions] = useState<ModelVersionItem[]>([]);
  const [experiments, setExperiments] = useState<ExperimentItem[]>([]);
  const [runs, setRuns] = useState<Record<string, RunItem[]>>({});
  const [loading, setLoading] = useState(true);
  const [fatalError, setFatalError] = useState<string | null>(null);
  const [promoting, setPromoting] = useState<string | null>(null);
  const [promoteMessage, setPromoteMessage] = useState<string | null>(null);
  const [promoteError, setPromoteError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);

  async function syncBackend() {
    setSyncing(true);
    setSyncMessage(null);
    setSyncError(null);
    try {
      const r = await ModelsApiService.syncBackend();
      setSyncMessage(`Backend synced to ${r.version} (${r.loaded_model})`);
    } catch (e) {
      setSyncError(e instanceof Error ? e.message : String(e));
    } finally {
      setSyncing(false);
    }
  }

  async function refreshModelsAndLatest() {
    const { models } = await ModelsApiService.listModels();
    setModelVersions(dedupeModelVersions(models));

    setLatest(null);
    setLatestError(null);
    try {
      const l = await ModelsApiService.getLatestModel();
      setLatest(l);
    } catch (e) {
      setLatestError(e instanceof Error ? e.message : String(e));
    }
  }

  async function promoteModel(model: ModelVersionItem) {
    const key = `${model.name}:${model.version}`;
    setPromoting(key);
    setPromoteMessage(null);
    setPromoteError(null);
    try {
      const promoted = await ModelsApiService.promoteModel(model.name, model.version, {
        alias: "champion",
        set_stage: true,
        stage: "Production",
        archive_existing_versions: true,
      });
      setPromoteMessage(
        `${promoted.name} v${promoted.version} promoted as ${promoted.alias}`
      );
      await refreshModelsAndLatest();
    } catch (e) {
      setPromoteError(e instanceof Error ? e.message : String(e));
    } finally {
      setPromoting(null);
    }
  }

  useEffect(() => {
    (async () => {
      try {
        const h = await ModelsApiService.getHealth();
        setHealth(h);

        const { experiments: exps } = await ModelsApiService.listExperiments();
        const uniqueExperiments = dedupeExperiments(exps);
        setExperiments(uniqueExperiments);

        const runsByExp: Record<string, RunItem[]> = {};
        await Promise.all(
          uniqueExperiments.map(async (e) => {
            const r = await ModelsApiService.listRuns(e.experiment_id);
            runsByExp[e.experiment_id] = dedupeRuns(r.runs);
          })
        );
        setRuns(runsByExp);

        await refreshModelsAndLatest();
      } catch (e) {
        setFatalError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <h1 style={styles.title}>Models & Experiments</h1>
        <Link to="/" style={styles.backLink}>
          ← Back
        </Link>
      </div>

      {fatalError && (
        <div style={styles.error}>Failed to reach model-api: {fatalError}</div>
      )}

      {/* Health */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>API Health</h2>
        {loading && !health ? (
          <span style={styles.muted}>Checking…</span>
        ) : health ? (
          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            <span style={styles.statusBadge(health.status === "ok")}>
              {health.status.toUpperCase()}
            </span>
            <span style={styles.muted}>
              MLflow: {health.mlflow_reachable ? "reachable" : "unreachable"}
            </span>
          </div>
        ) : null}
      </div>

      {/* Latest Production Model */}
      <div style={styles.section}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
          <h2 style={{ ...styles.sectionTitle, margin: 0 }}>Latest Production Model</h2>
          {latest && (
            <button
              style={{ ...styles.downloadButton, padding: "6px 14px", fontSize: "12px" }}
              disabled={syncing}
              onClick={syncBackend}
            >
              {syncing ? "Syncing..." : "Sync to Backend"}
            </button>
          )}
        </div>
        {syncMessage && <div style={{ ...styles.muted, marginBottom: "8px" }}>{syncMessage}</div>}
        {syncError && <div style={{ ...styles.error, marginBottom: "8px" }}>Sync failed: {syncError}</div>}
        {loading ? (
          <span style={styles.muted}>Loading…</span>
        ) : latestError ? (
          <div style={styles.muted}>
            {latestError.includes("404")
              ? "No model promoted as champion/Production yet."
              : latestError}
          </div>
        ) : latest ? (
          <>
            <div style={styles.grid}>
              <Card styles={styles} label="Version" value={latest.version} />
              <Card styles={styles} label="Trained" value={latest.trained} />
              <Card styles={styles} label="Task" value={latest.task} />
              <Card styles={styles} label="Config" value={latest.config} />
              <Card
                styles={styles}
                label="mAP50"
                value={latest.metrics.mAP50.toFixed(4)}
              />
              <Card
                styles={styles}
                label="mAP50-95"
                value={latest.metrics["mAP50-95"].toFixed(4)}
              />
              <Card
                styles={styles}
                label="Precision"
                value={latest.metrics.precision.toFixed(4)}
              />
              <Card
                styles={styles}
                label="Recall"
                value={latest.metrics.recall.toFixed(4)}
              />
            </div>
            <div style={{ marginTop: "16px" }}>
              <h3 style={{ ...styles.sectionTitle, fontSize: "14px" }}>
                Download
              </h3>
              <div style={styles.downloadRow}>
                <a
                  style={styles.downloadButton}
                  href={latest.assets.fp16}
                  download
                >
                  ⬇ best_fp16.onnx
                </a>
                <a
                  style={styles.downloadButton}
                  href={latest.assets.fp32}
                  download
                >
                  ⬇ best_fp32.onnx
                </a>
                <a
                  style={styles.downloadButton}
                  href={latest.assets.pt}
                  download
                >
                  ⬇ best.pt
                </a>
              </div>
            </div>
          </>
        ) : null}
      </div>

      {/* Registered Models */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>Registered Models</h2>
        {promoteMessage && <div style={styles.muted}>{promoteMessage}</div>}
        {promoteError && <div style={styles.error}>Promote failed: {promoteError}</div>}
        {loading ? (
          <span style={styles.muted}>Loading...</span>
        ) : modelVersions.length === 0 ? (
          <span style={styles.muted}>No registered model versions yet.</span>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Model</th>
                <th style={styles.th}>Version</th>
                <th style={styles.th}>Stage</th>
                <th style={styles.th}>Aliases</th>
                <th style={styles.th}>Run</th>
                <th style={styles.th}>Action</th>
              </tr>
            </thead>
            <tbody>
              {modelVersions.map((model) => {
                const key = `${model.name}:${model.version}`;
                const isChampion = model.aliases.includes("champion");
                return (
                  <tr key={key}>
                    <td style={styles.td}>{model.name}</td>
                    <td style={styles.td}>
                      <span style={styles.code}>v{model.version}</span>
                    </td>
                    <td style={styles.td}>{model.stage}</td>
                    <td style={styles.td}>
                      {model.aliases.length > 0 ? model.aliases.join(", ") : "-"}
                    </td>
                    <td style={styles.td}>
                      {model.run_id ? (
                        <span style={styles.code}>{model.run_id.slice(0, 8)}</span>
                      ) : (
                        <span style={styles.muted}>-</span>
                      )}
                    </td>
                    <td style={styles.td}>
                      <button
                        style={{ ...styles.downloadButton, padding: "4px 10px", fontSize: "11px" }}
                        disabled={isChampion || promoting === key}
                        onClick={() => promoteModel(model)}
                      >
                        {isChampion ? "Promoted" : promoting === key ? "Promoting..." : "Promote"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Experiments + Runs */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>Experiments</h2>
        {loading ? (
          <span style={styles.muted}>Loading…</span>
        ) : experiments.length === 0 ? (
          <span style={styles.muted}>No experiments yet.</span>
        ) : (
          experiments.map((exp) => (
            <div key={exp.experiment_id} style={{ marginBottom: "20px" }}>
              <h3 style={{ ...styles.sectionTitle, fontSize: "14px" }}>
                {exp.name}{" "}
                <span style={styles.code}>#{exp.experiment_id}</span>
              </h3>
              {(runs[exp.experiment_id] ?? []).length === 0 ? (
                <span style={styles.muted}>No runs.</span>
              ) : (
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>Run</th>
                      <th style={styles.th}>Status</th>
                      <th style={styles.th}>mAP50</th>
                      <th style={styles.th}>mAP50-95</th>
                      <th style={styles.th}>Release</th>
                      <th style={styles.th}>Download</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(runs[exp.experiment_id] ?? []).map((r) => {
                      const tag = r.tags.github_release_tag;
                      return (
                        <tr key={r.run_id}>
                          <td style={styles.td}>
                            {r.run_name ?? (
                              <span style={styles.code}>
                                {r.run_id.slice(0, 8)}
                              </span>
                            )}
                          </td>
                          <td style={styles.td}>{r.status}</td>
                          <td style={styles.td}>
                            {r.metrics.mAP50?.toFixed(4) ?? "—"}
                          </td>
                          <td style={styles.td}>
                            {r.metrics["mAP50-95"]?.toFixed(4) ?? "—"}
                          </td>
                          <td style={styles.td}>{tag ?? "—"}</td>
                          <td style={styles.td}>
                            {tag ? (
                              <div style={{ display: "flex", gap: "4px" }}>
                                <button
                                  style={{ ...styles.downloadButton, padding: "4px 10px", fontSize: "11px" }}
                                  onClick={() => downloadFromTag(tag, "fp16")}
                                >
                                  fp16
                                </button>
                                <button
                                  style={{ ...styles.downloadButton, padding: "4px 10px", fontSize: "11px" }}
                                  onClick={() => downloadFromTag(tag, "fp32")}
                                >
                                  fp32
                                </button>
                                <button
                                  style={{ ...styles.downloadButton, padding: "4px 10px", fontSize: "11px" }}
                                  onClick={() => downloadFromTag(tag, "pt")}
                                >
                                  pt
                                </button>
                              </div>
                            ) : (
                              <span style={styles.muted}>—</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function dedupeExperiments(experiments: ExperimentItem[]) {
  return Array.from(
    new Map(experiments.map((exp) => [exp.name, exp])).values()
  );
}

function dedupeRuns(runs: RunItem[]) {
  const seen = new Set<string>();
  return runs.filter((run) => {
    const key = run.tags.github_release_tag ?? run.run_name ?? run.run_id;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function dedupeModelVersions(models: ModelVersionItem[]) {
  return Array.from(
    new Map(models.map((model) => [`${model.name}:${model.version}`, model])).values()
  );
}

function Card({
  styles,
  label,
  value,
}: {
  styles: ReturnType<typeof getModelsViewStyles>;
  label: string;
  value: string;
}) {
  return (
    <div style={styles.card}>
      <div style={styles.cardLabel}>{label}</div>
      <div style={styles.cardValue}>{value}</div>
    </div>
  );
}
