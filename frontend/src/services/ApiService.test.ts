import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiService } from "./ApiService";
import * as configModule from "./config";

describe("ApiService custom pipeline methods", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.spyOn(configModule, "getApiUrl").mockResolvedValue("http://127.0.0.1:3005");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("createProject calls POST /api/projects with name and organism", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        project: { id: "proj-123", name: "Anolis Project", organism: "Anolis" },
      }),
    });
    globalThis.fetch = mockFetch;

    const project = await ApiService.createProject("Anolis Project", "Anolis");

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("http://127.0.0.1:3005/api/projects");
    expect(mockFetch.mock.calls[0][1].method).toBe("POST");
    expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toEqual({
      name: "Anolis Project",
      organism: "Anolis",
    });
    expect(project.id).toBe("proj-123");
  });

  it("listProjects calls GET /api/projects", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        projects: [{ id: "proj-1", name: "P1", organism: "O1" }],
      }),
    });
    globalThis.fetch = mockFetch;

    const projects = await ApiService.listProjects();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("http://127.0.0.1:3005/api/projects");
    expect(projects).toHaveLength(1);
  });

  it("getModels calls GET /api/models", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        models: [
          {
            id: "m-1",
            project_id: "p-1",
            name: "Model 1",
            created_at: "2026-08-31T00:00:00Z",
            manifest: {
              schema_version: 1,
              id: "m-1",
              name: "Model 1",
              description: "Test Model",
              detector: { artifact: "", geometry: "obb", confidence: 0.25, iou: 0.45 },
              classes: [],
              landmark_schemas: {},
            },
          },
        ],
      }),
    });
    globalThis.fetch = mockFetch;

    const models = await ApiService.getModels();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("http://127.0.0.1:3005/api/models");
    expect(models).toHaveLength(1);
    expect(models[0].id).toBe("m-1");
  });

  it("deleteModel calls DELETE /api/models/:id", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ success: true }),
    });
    globalThis.fetch = mockFetch;

    await ApiService.deleteModel("custom-model-1");
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("http://127.0.0.1:3005/api/models/custom-model-1");
    expect(mockFetch.mock.calls[0][1].method).toBe("DELETE");
  });

  it("submitTrain sends FormData with project_id and config", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, job_id: "job-abc" }),
    });
    globalThis.fetch = mockFetch;

    const file = new File(["dummy content"], "dataset.xml", { type: "text/xml" });
    const res = await ApiService.submitTrain("proj-1", [file], { epochs: 10, nu: 0.1 });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("http://127.0.0.1:3005/api/train");
    expect(mockFetch.mock.calls[0][1].method).toBe("POST");
    expect(res.job_id).toBe("job-abc");
  });

  it("getTrainJobStatus fetches status from /api/train/:jobId", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        job_id: "job-abc",
        status: "running",
        stage: "Epoch 3/10",
        progress: 0.3,
        metrics: {},
      }),
    });
    globalThis.fetch = mockFetch;

    const status = await ApiService.getTrainJobStatus("job-abc");
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("http://127.0.0.1:3005/api/train/job-abc");
    expect(status.status).toBe("running");
  });

  it("cancelTrainJob calls POST /api/train/:jobId/cancel", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, message: "Cancelled" }),
    });
    globalThis.fetch = mockFetch;

    const res = await ApiService.cancelTrainJob("job-abc");
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("http://127.0.0.1:3005/api/train/job-abc/cancel");
    expect(res.success).toBe(true);
  });
});
