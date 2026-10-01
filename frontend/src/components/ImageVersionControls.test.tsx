import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { ImageVersionControls } from "./ImageVersionControls";

const render = (overrides: Partial<React.ComponentProps<typeof ImageVersionControls>> = {}) =>
  renderToStaticMarkup(
    <ImageVersionControls
      dataFetched={true}
      imageSet={{ original: "/o.jpg", inverted: "/i.jpg", color_contrasted: "/c.jpg" }}
      currentImageURL="/o.jpg"
      loading={false}
      dataLoading={false}
      onVersionChange={() => {}}
      isEditMode={false}
      onToggleEditMode={() => {}}
      onResetZoom={() => {}}
      theme="light"
      {...overrides}
    />
  );

describe("ImageVersionControls toepad grid toggle", () => {
  it("is hidden outside the toepad view", () => {
    expect(render()).not.toContain("Toepad Grid");
  });

  it("shows the off state by default", () => {
    const markup = render({ onToggleToepadGrid: () => {} });

    expect(markup).toContain("Toepad Grid: Off");
    expect(markup).toContain('aria-pressed="false"');
  });

  it("shows the on state when the grid is on", () => {
    const markup = render({ onToggleToepadGrid: () => {}, isToepadGridOn: true });

    expect(markup).toContain("Toepad Grid: On");
    expect(markup).toContain('aria-pressed="true"');
  });
});
