import * as d3 from "d3";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiService } from "../services/ApiService";
import { SessionService } from "../services/SessionService";
import type { ToepadRegion } from "../services/ToepadRegions";
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

  it("switches directly from one toepad to another without resetting to full view", () => {
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

    const toe1 = {
      slot: { label: "up_finger" as const, title: "Upper Finger", color: "#00cc00" },
      crop: { x: 100, y: 100, size: 80 },
      points: [],
    };
    const toe2 = {
      slot: { label: "bot_toe" as const, title: "Lower Toe", color: "#0088ff" },
      crop: { x: 500, y: 400, size: 90 },
      points: [],
    };

    const handler = (view as unknown as { handleZoomToToe: (r: ToepadRegion) => void }).handleZoomToToe;

    // Zoom into toe 1
    handler(toe1);
    expect(view.state.activeToepadSlotLabel).toBe("up_finger");
    const transform1 = view.state.zoomTransform;

    // Zoom into toe 2 directly
    handler(toe2);
    expect(view.state.activeToepadSlotLabel).toBe("bot_toe");
    const transform2 = view.state.zoomTransform;
    expect(transform2).not.toEqual(d3.zoomIdentity);
    expect(transform2).not.toEqual(transform1);
  });

  it("delegates to svgViewerRef.zoomToImageRegion when svgViewerRef is available", () => {
    const view = new MainView({ selectedViewType: "toepads" });
    const expectedTransform = d3.zoomIdentity.translate(200, 150).scale(4.5);
    const zoomSpy = vi.fn().mockReturnValue(expectedTransform);

    (view.svgViewerRef as unknown as { current: unknown }).current = {
      zoomToImageRegion: zoomSpy,
    };

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

    expect(zoomSpy).toHaveBeenCalledWith(region.crop);
    expect(view.state.activeToepadSlotLabel).toBe("up_finger");
    expect(view.state.zoomTransform).toBe(expectedTransform);
  });

  it("ignores handleZoomToToe when region has null crop", () => {
    const view = new MainView({ selectedViewType: "toepads" });
    const initialTransform = d3.zoomIdentity;
    view.state.zoomTransform = initialTransform;
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

    const undetectedRegion = {
      slot: { label: "up_finger" as const, title: "Upper Finger", color: "#00cc00" },
      crop: null,
      points: [],
    };

    (view as unknown as { handleZoomToToe: (r: typeof undetectedRegion) => void }).handleZoomToToe(undetectedRegion);

    expect(view.state.activeToepadSlotLabel).toBeNull();
    expect(view.state.zoomTransform).toBe(initialTransform);
  });

  it("resets activeToepadSlotLabel when changing image via changeCurrentImage", () => {
    const view = new MainView({ selectedViewType: "toepads" });
    view.state.images = [
      {
        name: "img1.png",
        imageSets: { original: "/img1.png", inverted: "", color_contrasted: "" },
        coords: [],
        originalCoords: [],
        boundingBoxes: [],
      },
      {
        name: "img2.png",
        imageSets: { original: "/img2.png", inverted: "", color_contrasted: "" },
        coords: [],
        originalCoords: [],
        boundingBoxes: [],
      },
    ] as any;
    view.state.currentImageIndex = 0;
    view.state.activeToepadSlotLabel = "up_finger";

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

    (view as unknown as { changeCurrentImage: (idx: number) => void }).changeCurrentImage(1);

    expect(view.state.currentImageIndex).toBe(1);
    expect(view.state.activeToepadSlotLabel).toBeNull();
  });

  it("resets activeToepadSlotLabel and zoomTransform on executeClearHistory", async () => {
    vi.spyOn(ApiService, "clearHistory").mockResolvedValue(undefined as any);
    vi.spyOn(SessionService, "clearSession").mockReturnValue();
    vi.spyOn(SessionService, "initializeSession").mockResolvedValue(undefined as any);

    const view = new MainView({ selectedViewType: "toepads" });
    view.state.activeToepadSlotLabel = "up_finger";
    view.state.zoomTransform = d3.zoomIdentity.translate(50, 50).scale(3);

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

    await (view as unknown as { executeClearHistory: () => Promise<void> }).executeClearHistory();

    expect(view.state.activeToepadSlotLabel).toBeNull();
    expect(view.state.zoomTransform).toEqual(d3.zoomIdentity);
  });
});

describe("MainView Specimen Review & Verification Status", () => {
  const createMockView = () => {
    const view = new MainView({ selectedViewType: "toepads" });
    const img1 = {
      name: "specimen_1.jpg",
      coords: [{ id: 0, x: 10, y: 10 }],
      originalCoords: [{ id: 0, x: 10, y: 10 }],
      imageSets: { original: "img1.png", inverted: "", color_contrasted: "" },
      timestamp: "12:00",
      reviewStatus: "unreviewed" as const,
    };
    const img2 = {
      name: "specimen_2.jpg",
      coords: [{ id: 0, x: 20, y: 20 }],
      originalCoords: [{ id: 0, x: 20, y: 20 }],
      imageSets: { original: "img2.png", inverted: "", color_contrasted: "" },
      timestamp: "12:05",
      reviewStatus: "needs_review" as const,
    };

    view.state.images = [img1, img2];
    view.state.uploadHistory = [
      { name: "specimen_1.jpg", timestamp: "12:00", index: 0, viewType: "toepads", reviewStatus: "unreviewed" },
      { name: "specimen_2.jpg", timestamp: "12:05", index: 1, viewType: "toepads", reviewStatus: "needs_review" },
    ];
    view.state.currentImageIndex = 0;
    view.state.imageFilename = "specimen_1.jpg";
    view.state.currentImageURL = "img1.png";

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

    return view;
  };

  it("handleApproveAndNext marks current image as verified and advances to next image", () => {
    const view = createMockView();
    expect(view.state.images[0].reviewStatus).toBe("unreviewed");

    view.handleApproveAndNext();

    expect(view.state.images[0].reviewStatus).toBe("verified");
    expect(view.state.uploadHistory[0].reviewStatus).toBe("verified");
    // Advanced to next image (specimen_2)
    expect(view.state.currentImageIndex).toBe(1);
    expect(view.state.imageFilename).toBe("specimen_2.jpg");
  });

  it("handleApproveAndNext marks the last image as verified and stays on it", () => {
    const view = createMockView();
    view.state.currentImageIndex = 1;
    view.state.imageFilename = "specimen_2.jpg";

    view.handleApproveAndNext();

    expect(view.state.images[1].reviewStatus).toBe("verified");
    expect(view.state.uploadHistory[1].reviewStatus).toBe("verified");
    expect(view.state.currentImageIndex).toBe(1);
  });

  it("handleChangeReviewStatus updates review status for current image and upload history", () => {
    const view = createMockView();
    view.handleChangeReviewStatus("needs_review");

    expect(view.state.images[0].reviewStatus).toBe("needs_review");
    expect(view.state.uploadHistory[0].reviewStatus).toBe("needs_review");
  });

  it("handleScatterDataUpdate transitions verified image back to needs_review when landmarks change", () => {
    const view = createMockView();
    view.state.images[0].reviewStatus = "verified";

    // Landmark edited
    (view as any).handleScatterDataUpdate([{ id: 0, x: 99, y: 99 }], [{ id: 0, x: 99, y: 99 }]);

    expect(view.state.images[0].reviewStatus).toBe("needs_review");
    expect(view.state.uploadHistory[0].reviewStatus).toBe("needs_review");
  });

  it("handleToepadPointsChange transitions verified image back to needs_review when toepad points change", () => {
    const view = createMockView();
    view.state.images[0].reviewStatus = "verified";

    (view as any).handleToepadPointsChange([{ id: 0, x: 55, y: 55 }]);

    expect(view.state.images[0].reviewStatus).toBe("needs_review");
    expect(view.state.uploadHistory[0].reviewStatus).toBe("needs_review");
  });

  it("handleGlobalKeyDown triggers handleApproveAndNext on Space or Enter", () => {
    const view = createMockView();
    const spy = vi.spyOn(view, "handleApproveAndNext");

    const preventDefault = vi.fn();
    view.handleGlobalKeyDown({ key: " ", preventDefault, target: { tagName: "div" } } as any);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(preventDefault).toHaveBeenCalled();

    view.handleGlobalKeyDown({ key: "Enter", preventDefault, target: { tagName: "div" } } as any);
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("handleGlobalKeyDown ignores shortcut when in edit mode or inside an input", () => {
    const view = createMockView();
    const spy = vi.spyOn(view, "handleApproveAndNext");

    // In edit mode
    view.state.isEditMode = true;
    view.handleGlobalKeyDown({ key: " ", preventDefault: vi.fn(), target: { tagName: "div" } } as any);
    expect(spy).not.toHaveBeenCalled();

    // Inside an input element
    view.state.isEditMode = false;
    view.handleGlobalKeyDown({ key: "Enter", preventDefault: vi.fn(), target: { tagName: "input" } } as any);
    expect(spy).not.toHaveBeenCalled();
  });
});
