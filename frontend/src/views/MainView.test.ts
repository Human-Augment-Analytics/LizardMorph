import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiService } from "../services/ApiService";
import { MainView } from "./MainView";

describe("MainView free predictor panel", () => {
  it("opens the free-mode predictor panel by default", () => {
    const view = new MainView({ selectedViewType: "free" });

    expect(view.state.isFreePredictorPanelOpen).toBe(true);
  });

  it("keeps the predictor panel collapsed for non-free modes", () => {
    const view = new MainView({ selectedViewType: "dorsal" });

    expect(view.state.isFreePredictorPanelOpen).toBe(false);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads available predictors during initial free-mode startup", async () => {
    vi.spyOn(ApiService, "initialize").mockResolvedValue();
    const listPredictors = vi.spyOn(ApiService, "listPredictors").mockResolvedValue([
      {
        id: "predictor-1",
        display_name: "demo_predictor.dat",
        stored_filename: "demo_predictor.dat",
        num_parts: 12,
        uploaded_at: "2026-03-26T00:00:00Z",
      },
    ]);

    const view = new MainView({ selectedViewType: "free" });
    (view as unknown as { fetchUploadedFiles: () => void }).fetchUploadedFiles = vi.fn();
    (view as unknown as { setupBeforeUnloadHandler: () => void }).setupBeforeUnloadHandler = vi.fn();
    view.setState = ((update: unknown) => {
      const nextState =
        typeof update === "function"
          ? update(view.state, view.props)
          : update;

      view.state = {
        ...view.state,
        ...(nextState as Partial<typeof view.state>),
      };
    }) as typeof view.setState;

    await (view as never as { initializeApp: () => Promise<void> }).initializeApp();

    expect(listPredictors).toHaveBeenCalledTimes(1);
    expect(view.state.availablePredictors).toHaveLength(1);
    expect(view.state.availablePredictors[0]?.id).toBe("predictor-1");
  });
});

describe("MainView upload history", () => {
  afterEach(() => vi.restoreAllMocks());

  it("does not auto-load an image that an active upload is already processing", async () => {
    const file = new File(["image"], "specimen.jpg", { type: "image/jpeg" });
    let finishList!: (files: { filename: string; view_type: string }[]) => void;
    vi.spyOn(ApiService, "fetchUploadedFiles").mockImplementation(
      () => new Promise((resolve) => { finishList = resolve; })
    );
    let finishUpload!: (images: { name: string; coords: [] }[]) => void;
    vi.spyOn(ApiService, "uploadMultipleImages").mockImplementation(
      () => new Promise((resolve) => { finishUpload = resolve; })
    );
    vi.spyOn(ApiService, "fetchImageSet").mockResolvedValue({
      original: "/specimen.jpg",
      inverted: "/inverted.jpg",
      color_contrasted: "/contrast.jpg",
    });
    const processExisting = vi.spyOn(ApiService, "processExistingImage").mockResolvedValue({
      name: file.name,
      coords: [],
    });

    const view = new MainView({ selectedViewType: "free" });
    view.setState = ((update: unknown, callback?: () => void) => {
      const nextState = typeof update === "function" ? update(view.state, view.props) : update;
      view.state = { ...view.state, ...nextState };
      callback?.();
    }) as typeof view.setState;
    const editor = view as unknown as {
      fetchUploadedFiles: () => Promise<void>;
      handleUpload: (event: { target: { files: File[] } }) => Promise<void>;
      loadImageFromUploads: (filename: string) => Promise<void>;
    };

    const history = editor.fetchUploadedFiles();
    await editor.handleUpload({ target: { files: [file] } });
    finishList([{ filename: file.name, view_type: "free" }]);
    await history;
    await editor.loadImageFromUploads(file.name);
    finishUpload([{ name: file.name, coords: [] }]);
    await vi.waitFor(() => expect(view.state.loading).toBe(false));

    expect(processExisting).not.toHaveBeenCalled();
    expect(view.state.images.map((image) => image.name)).toEqual([file.name]);
    expect(view.state.uploadHistory.map((image) => image.name)).toEqual([file.name]);
  });

  function createEditor() {
    const view = new MainView({ selectedViewType: "free" });
    view.setState = ((update: unknown, callback?: () => void) => {
      const nextState = typeof update === "function" ? update(view.state, view.props) : update;
      view.state = { ...view.state, ...nextState };
      callback?.();
    }) as typeof view.setState;
    const editor = view as unknown as {
      handleUpload: (event: { target: { files: File[] } }) => Promise<void>;
      loadImageFromUploads: (filename: string) => Promise<void>;
    };
    return { view, editor };
  }

  it("keeps a batch busy until every upload settles after an early failure", async () => {
    const failed = new File(["a"], "failed.jpg", { type: "image/jpeg" });
    const pending = new File(["b"], "pending.jpg", { type: "image/jpeg" });
    let finishPending!: (images: { name: string; coords: [] }[]) => void;
    vi.spyOn(ApiService, "uploadMultipleImages").mockImplementation((files) =>
      files[0].name === failed.name
        ? Promise.reject(new Error("failed.jpg rejected"))
        : new Promise((resolve) => { finishPending = resolve; })
    );
    vi.spyOn(ApiService, "fetchImageSet").mockResolvedValue({
      original: "/pending.jpg",
      inverted: "/inverted.jpg",
      color_contrasted: "/contrast.jpg",
    });
    const processExisting = vi.spyOn(ApiService, "processExistingImage").mockResolvedValue({
      name: pending.name,
      coords: [],
    });
    vi.spyOn(console, "error").mockImplementation(() => {});
    const { view, editor } = createEditor();

    await editor.handleUpload({ target: { files: [failed, pending] } });
    await vi.waitFor(() => expect(view.state.uploadProgress[failed.name]).toBe(-1));

    expect(view.state.loading).toBe(true);
    expect(view.state.uploadProgress[pending.name]).toBeDefined();
    await editor.loadImageFromUploads(pending.name);
    expect(processExisting).not.toHaveBeenCalled();

    finishPending([{ name: pending.name, coords: [] }]);
    await vi.waitFor(() => expect(view.state.loading).toBe(false));

    expect(view.state.dataError?.message).toBe("failed.jpg rejected");
    expect(view.state.uploadProgress).toEqual({});
    expect(view.state.images.map((image) => image.name)).toEqual([pending.name]);
  });

  it("does not replace an already loaded image when its upload finishes later", async () => {
    const file = new File(["image"], "specimen.jpg", { type: "image/jpeg" });
    const edited = [{ id: 1, x: 40, y: 50 }];
    vi.spyOn(ApiService, "uploadMultipleImages").mockResolvedValue([
      { name: file.name, coords: [{ id: 1, x: 1, y: 2 }] },
    ]);
    vi.spyOn(ApiService, "fetchImageSet").mockResolvedValue({
      original: "/specimen.jpg",
      inverted: "/inverted.jpg",
      color_contrasted: "/contrast.jpg",
    });
    const { view, editor } = createEditor();
    view.state = {
      ...view.state,
      images: [{
        name: file.name,
        coords: edited,
        originalCoords: edited,
        imageSets: { original: "/specimen.jpg", inverted: "", color_contrasted: "" },
        timestamp: "From uploads folder",
      }],
    };

    await editor.handleUpload({ target: { files: [file] } });
    await vi.waitFor(() => expect(view.state.loading).toBe(false));

    expect(view.state.images).toHaveLength(1);
    expect(view.state.images[0].coords).toEqual(edited);
  });
});
