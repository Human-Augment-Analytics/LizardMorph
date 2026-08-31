import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainView } from "./TrainView";
import { ApiService } from "../services/ApiService";

describe("TrainView Wizard Component", () => {
  beforeEach(() => {
    vi.spyOn(ApiService, "getModels").mockResolvedValue([]);
    vi.spyOn(ApiService, "createProject").mockResolvedValue({
      id: "proj-1",
      name: "Anolis Morphometrics",
      organism: "Anolis carolinensis",
      created_at: "2026-08-31T00:00:00Z",
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the 7-step wizard navigation and step 1 inputs", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <TrainView
          onNavigateHome={() => {}}
          onUseModel={() => {}}
        />
      </MemoryRouter>
    );

    // Header & Steps
    expect(markup).toContain("Model Training Wizard");
    expect(markup).toContain("Create Project");
    expect(markup).toContain("Add Data");
    expect(markup).toContain("Check Data");
    expect(markup).toContain("Train Model");
    expect(markup).toContain("Follow Progress");
    expect(markup).toContain("Review Results");
    expect(markup).toContain("Publish &amp; Use");

    // Step 1 Form Fields
    expect(markup).toContain("Project Name");
    expect(markup).toContain("Target Organism / Taxon");
    expect(markup).toContain("Anolis Morphometrics");
    expect(markup).toContain("Continue to Add Data →");
  });
});
