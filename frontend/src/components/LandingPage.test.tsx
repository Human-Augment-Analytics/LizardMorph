import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LandingPage } from "./LandingPage";
import { ApiService } from "../services/ApiService";

describe("LandingPage Component", () => {
  beforeEach(() => {
    vi.spyOn(ApiService, "getModels").mockResolvedValue([
      {
        id: "custom-wing-model",
        project_id: "proj-wing",
        name: "Drosophila Wing v1",
        version: 1,
        organism: "Drosophila melanogaster",
        description: "Wing landmarks detector",
        created_at: "2026-08-31T00:00:00Z",
      },
    ]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the AutoMorph hero title and preinstalled modes", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <LandingPage
          onSelectMode={() => {}}
          onOpenTrainWizard={() => {}}
        />
      </MemoryRouter>
    );

    expect(markup).toContain("AutoMorph");
    expect(markup).toContain("Select a preinstalled lizard model");
    expect(markup).toContain("Dorsal View");
    expect(markup).toContain("Lateral View");
    expect(markup).toContain("Toepad View");
    expect(markup).toContain("Free Mode");
  });

  it("renders the Train Custom Model action card", () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <LandingPage
          onSelectMode={() => {}}
          onOpenTrainWizard={() => {}}
        />
      </MemoryRouter>
    );

    expect(markup).toContain("Want to create a custom project model for your species?");
    expect(markup).toContain("Train Custom Model Wizard");
  });
});
