import { Component } from "react";
import type React from "react";
import type { Point } from "../models/Point";
import type { BoundingBox } from "../models/AnnotationsData";
import type { ResolvedTheme } from "../contexts/theme";
import { getToepadRegions } from "../services/ToepadRegions";
import type { ToepadRegion } from "../services/ToepadRegions";
import { getToepadPanelsStyles } from "./ToepadPanels.style";

interface ToepadPanelsProps {
  imageURL: string | null;
  imageWidth: number;
  imageHeight: number;
  /** All landmarks for the image, in image-pixel space. */
  points: Point[];
  boundingBoxes: BoundingBox[];
  selectedPoint: Point | null;
  isEditMode: boolean;
  onToggleEditMode: () => void;
  onPointSelect: (point: Point | null) => void;
  onPointsChange: (points: Point[]) => void;
  theme: ResolvedTheme;
}

interface DragState {
  id: number;
  x: number;
  y: number;
  startClientX: number;
  startClientY: number;
  moved: boolean;
}

interface ToepadPanelsState {
  drag: DragState | null;
}

const CLICK_THRESHOLD = 3;

export class ToepadPanels extends Component<ToepadPanelsProps, ToepadPanelsState> {
  state: ToepadPanelsState = {
    drag: null,
  };

  private readonly toImagePoint = (
    svg: SVGSVGElement,
    clientX: number,
    clientY: number
  ): { x: number; y: number } | null => {
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const pt = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
    return {
      x: Math.min(Math.max(pt.x, 0), this.props.imageWidth),
      y: Math.min(Math.max(pt.y, 0), this.props.imageHeight),
    };
  };

  private readonly handlePointPointerDown = (
    event: React.PointerEvent<SVGGElement>,
    point: Point
  ): void => {
    if (!this.props.isEditMode || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
    if (this.props.selectedPoint?.id !== point.id) {
      this.props.onPointSelect(point);
    }
    this.setState({
      drag: {
        id: point.id,
        x: point.x,
        y: point.y,
        startClientX: event.clientX,
        startClientY: event.clientY,
        moved: false,
      },
    });
  };

  private readonly handlePointerMove = (event: React.PointerEvent<SVGSVGElement>): void => {
    const { drag } = this.state;
    if (!drag) return;
    const moved =
      drag.moved ||
      Math.hypot(event.clientX - drag.startClientX, event.clientY - drag.startClientY) >=
        CLICK_THRESHOLD;
    if (!moved) return;
    const pt = this.toImagePoint(event.currentTarget, event.clientX, event.clientY);
    if (!pt) return;
    this.setState({ drag: { ...drag, ...pt, moved: true } });
  };

  private readonly handlePointerUp = (event: React.PointerEvent<SVGSVGElement>): void => {
    const { drag } = this.state;
    if (!drag) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    this.setState({ drag: null });
    if (drag.moved) {
      this.props.onPointsChange(
        this.props.points.map((p) => (p.id === drag.id ? { ...p, x: drag.x, y: drag.y } : p))
      );
    }
  };

  private readonly handlePointerCancel = (): void => {
    this.setState({ drag: null });
  };

  private readonly handleContextMenu = (event: React.MouseEvent): void => {
    event.preventDefault();
    this.props.onToggleEditMode();
  };

  private renderRegion(region: ToepadRegion) {
    const { imageURL, imageWidth, imageHeight, selectedPoint, isEditMode, theme } = this.props;
    const { drag } = this.state;
    const styles = getToepadPanelsStyles(theme);
    const { slot, crop, points } = region;

    return (
      <div key={slot.label} style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle} title={slot.label}>
            <span style={{ ...styles.swatch, backgroundColor: slot.color }} />
            {slot.title}
          </span>
          {crop && <span style={styles.countChip}>{points.length} pts</span>}
        </div>
        <div
          style={{
            ...styles.viewport,
            ...(isEditMode && crop ? styles.viewportEditing : {}),
          }}
        >
          {!crop || !imageURL ? (
            <div style={styles.emptyState}>Not detected</div>
          ) : (
            <>
              <img
                src={imageURL}
                alt={`${slot.title} close-up`}
                draggable={false}
                style={{
                  ...styles.cropImage,
                  left: `${(-crop.x / crop.size) * 100}%`,
                  top: `${(-crop.y / crop.size) * 100}%`,
                  width: `${(imageWidth / crop.size) * 100}%`,
                  height: `${(imageHeight / crop.size) * 100}%`,
                }}
              />
              <svg
                viewBox={`${crop.x} ${crop.y} ${crop.size} ${crop.size}`}
                preserveAspectRatio="none"
                style={{ ...styles.overlay, cursor: isEditMode ? "crosshair" : "default" }}
                onPointerMove={this.handlePointerMove}
                onPointerUp={this.handlePointerUp}
                onPointerCancel={this.handlePointerCancel}
                onContextMenu={this.handleContextMenu}
              >
                <defs>
                  <filter
                    id={`toepad-label-shadow-${slot.label}`}
                    x="-20%"
                    y="-20%"
                    width="140%"
                    height="140%"
                  >
                    <feDropShadow
                      dx="0"
                      dy="0"
                      stdDeviation={crop.size * 0.006}
                      floodColor="black"
                      floodOpacity="0.9"
                    />
                  </filter>
                </defs>
                {points.map((p, i) => {
                  const isDragged = drag?.id === p.id && drag.moved;
                  const x = isDragged ? drag.x : p.x;
                  const y = isDragged ? drag.y : p.y;
                  const r = crop.size * 0.012;
                  const isSelected = selectedPoint?.id === p.id;
                  return (
                    <g
                      key={p.id}
                      onPointerDown={(e) => this.handlePointPointerDown(e, p)}
                      style={{ cursor: isEditMode ? (isDragged ? "grabbing" : "grab") : "default" }}
                    >
                      {isEditMode && <circle cx={x} cy={y} r={r * 3} fill="transparent" />}
                      <circle cx={x} cy={y} r={r} fill={isSelected ? "yellow" : "red"} />
                      <text
                        x={x + r * 1.5}
                        y={y - r * 1.5}
                        fontSize={crop.size * 0.036}
                        fill="#ffffff"
                        filter={`url(#toepad-label-shadow-${slot.label})`}
                        style={{ pointerEvents: "none", userSelect: "none" }}
                      >
                        {i + 1}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </>
          )}
        </div>
      </div>
    );
  }

  render() {
    const { imageURL, imageWidth, imageHeight, points, boundingBoxes, isEditMode, theme } = this.props;
    if (!imageURL || !imageWidth || !imageHeight) return null;

    const regions = getToepadRegions(points, boundingBoxes);
    const detected = regions.filter((r) => r.crop).length;
    const styles = getToepadPanelsStyles(theme);

    return (
      <div style={styles.container}>
        <div style={styles.header}>
          <span style={styles.headerTitle}>
            <span>Toepads</span>
            <span style={styles.headerSubtitle}>
              {detected} of {regions.length} detected
            </span>
          </span>
          <span style={styles.headerSubtitle}>
            {isEditMode ? "Drag a landmark to move it" : "Click Edit Points to adjust landmarks"}
          </span>
        </div>
        <div style={styles.grid}>{regions.map((r) => this.renderRegion(r))}</div>
      </div>
    );
  }
}
