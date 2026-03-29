import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { FreePredictorPanel } from "./FreePredictorPanel";

describe("FreePredictorPanel", () => {
  it("renders the predictor select with explicit readable text styling", () => {
    const markup = renderToStaticMarkup(
      <FreePredictorPanel
        predictors={[
          {
            id: "predictor-1",
            display_name: "demo_predictor.dat",
            stored_filename: "demo_predictor.dat",
            num_parts: 12,
            uploaded_at: "2026-03-26T00:00:00Z",
          },
        ]}
        selectedPredictorId="predictor-1"
        predictorsLoading={false}
        error={null}
        hasCurrentImage={true}
        onRefresh={() => {}}
        onSelectPredictorId={() => {}}
        onUploadPredictor={() => {}}
        onDeleteSelected={() => {}}
        onAutoplace={() => {}}
        theme="light"
      />
    );

    expect(markup).toContain("color:#111");
    expect(markup).toContain("color-scheme:light");
  });
});
