import * as d3 from "d3";
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

  it("starts the toepad view with the toepad grid off", () => {
    const view = new MainView({ selectedViewType: "toepads" });

    expect(view.state.isToepadGridOn).toBe(false);
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

describe("MainView history panel collapse", () => {
  it("defaults isHistoryCollapsed to false", () => {
    const view = new MainView({ selectedViewType: "toepads" });
    expect(view.state.isHistoryCollapsed).toBe(false);
  });

  it("toggles isHistoryCollapsed when handleToggleHistoryCollapse is invoked", () => {
    const view = new MainView({ selectedViewType: "toepads" });
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

    (view as unknown as { handleToggleHistoryCollapse: () => void }).handleToggleHistoryCollapse();
    expect(view.state.isHistoryCollapsed).toBe(true);

    (view as unknown as { handleToggleHistoryCollapse: () => void }).handleToggleHistoryCollapse();
    expect(view.state.isHistoryCollapsed).toBe(false);
  });
});

describe("MainView toepad zoom feature", () => {
  it("zooms into toepad region on handleZoomToToe", () => {
    const view = new MainView({ selectedViewType: "toepads" });
    view.state.imageWidth = 1000;
    view.state.imageHeight = 800;
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

    const region = {
      slot: { label: "up_finger" as const, title: "Upper Finger", color: "#00cc00" },
      crop: { x: 200, y: 150, size: 80 },
      points: [],
    };

    (view as unknown as { handleZoomToToe: (r: typeof region) => void }).handleZoomToToe(region);

    expect(view.state.activeToepadSlotLabel).toBe("up_finger");
    expect(view.state.zoomTransform).not.toEqual(d3.zoomIdentity);
    expect(view.state.zoomTransform.k).toBeGreaterThan(1);
  });

  it("toggles zoom off back to full view when clicking the same toepad twice", () => {
    const view = new MainView({ selectedViewType: "toepads" });
    view.state.imageWidth = 1000;
    view.state.imageHeight = 800;
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

    const region = {
      slot: { label: "up_finger" as const, title: "Upper Finger", color: "#00cc00" },
      crop: { x: 200, y: 150, size: 80 },
      points: [],
    };

    const handler = (view as unknown as { handleZoomToToe: (r: typeof region) => void }).handleZoomToToe;
    handler(region);
    expect(view.state.activeToepadSlotLabel).toBe("up_finger");

    // Second click on same toepad resets
    handler(region);
    expect(view.state.activeToepadSlotLabel).toBeNull();
    expect(view.state.zoomTransform).toEqual(d3.zoomIdentity);
  });

  it("handleResetZoom clears activeToepadSlotLabel and resets zoomTransform", () => {
    const view = new MainView({ selectedViewType: "toepads" });
    view.state.activeToepadSlotLabel = "up_finger";
    view.state.zoomTransform = d3.zoomIdentity.translate(100, 50).scale(3);
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

    (view as unknown as { handleResetZoom: () => void }).handleResetZoom();
    expect(view.state.activeToepadSlotLabel).toBeNull();
    expect(view.state.zoomTransform).toEqual(d3.zoomIdentity);
  });
});
