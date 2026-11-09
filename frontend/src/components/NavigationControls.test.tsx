import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { NavigationControls } from "./NavigationControls";

describe("NavigationControls", () => {
  it("renders previous, next, and image counter when multiple images exist", () => {
    const markup = renderToStaticMarkup(
      <NavigationControls
        currentImageIndex={0}
        totalImages={3}
        loading={false}
        onPrevious={() => {}}
        onNext={() => {}}
        theme="light"
      />
    );

    expect(markup).toContain("Previous Image");
    expect(markup).toContain("Next Image");
    expect(markup).toContain("Image 1 of 3");
  });

  it("renders status dropdown and approve button when handlers are provided", () => {
    const markup = renderToStaticMarkup(
      <NavigationControls
        currentImageIndex={0}
        totalImages={2}
        loading={false}
        onPrevious={() => {}}
        onNext={() => {}}
        currentReviewStatus="needs_review"
        onApproveAndNext={() => {}}
        onChangeReviewStatus={() => {}}
        theme="light"
      />
    );

    expect(markup).toContain("data-testid=\"qc-status-select\"");
    expect(markup).toContain("🟡 Needs Review");
    expect(markup).toContain("data-testid=\"approve-and-next-button\"");
    expect(markup).toContain("✓ Approve &amp; Next");
    expect(markup).toContain("Space");
  });

  it("disables navigation and approve button when loading is true", () => {
    const markup = renderToStaticMarkup(
      <NavigationControls
        currentImageIndex={0}
        totalImages={2}
        loading={true}
        onPrevious={() => {}}
        onNext={() => {}}
        onApproveAndNext={() => {}}
        onChangeReviewStatus={() => {}}
        theme="light"
      />
    );

    expect(markup).toContain("disabled");
  });

  it("triggers onApproveAndNext callback on button click", () => {
    const onApprove = vi.fn();
    const controls = new NavigationControls({
      currentImageIndex: 0,
      totalImages: 2,
      loading: false,
      onPrevious: () => {},
      onNext: () => {},
      currentReviewStatus: "unreviewed",
      onApproveAndNext: onApprove,
      onChangeReviewStatus: () => {},
      theme: "light",
    });

    const rendered = controls.render() as React.ReactElement<{
      children: Array<React.ReactElement<{
        "data-testid"?: string;
        onClick?: () => void;
      }>>;
    }>;

    const approveBtn = rendered.props.children.find(
      (c) => c && c.props && c.props["data-testid"] === "approve-and-next-button"
    );
    expect(approveBtn).toBeDefined();
    approveBtn?.props.onClick?.();
    expect(onApprove).toHaveBeenCalledTimes(1);
  });

  it("triggers onChangeReviewStatus when status dropdown changes", () => {
    const onChangeStatus = vi.fn();
    const controls = new NavigationControls({
      currentImageIndex: 0,
      totalImages: 2,
      loading: false,
      onPrevious: () => {},
      onNext: () => {},
      currentReviewStatus: "unreviewed",
      onApproveAndNext: () => {},
      onChangeReviewStatus: onChangeStatus,
      theme: "light",
    });

    const rendered = controls.render() as React.ReactElement<{
      children: Array<React.ReactElement<{
        children?: Array<React.ReactElement<{
          "data-testid"?: string;
          onChange?: (e: { target: { value: string } }) => void;
        }>>;
      }>>;
    }>;

    const group = rendered.props.children.find(
      (c) => c && c.props && Array.isArray(c.props.children) && c.props.children.some(
        (child) => child && child.props && child.props["data-testid"] === "qc-status-select"
      )
    );
    expect(group).toBeDefined();
    const select = group?.props.children?.find(
      (child) => child && child.props && child.props["data-testid"] === "qc-status-select"
    );
    select?.props.onChange?.({ target: { value: "verified" } });
    expect(onChangeStatus).toHaveBeenCalledWith("verified");
  });
});
