import { Component } from "react";
import type React from "react";
import type { Point } from "../models/Point";
import type { BoundingBox } from "../models/AnnotationsData";
import type { ResolvedTheme } from "../contexts/theme";
import { getToepadRegions } from "../services/ToepadRegions";
import type { ToepadRegion } from "../services/ToepadRegions";
import { getToepadPanelsStyles } from "./ToepadPanels.style";
import { ToepadModal } from "./ToepadModal";

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
  style?: React.CSSProperties;
  defaultLegendOpen?: boolean;
  defaultActiveModalSlotLabel?: string | null;
  /** Callback to zoom the big image in onto a specific toepad region */
  onZoomToToe?: (region: ToepadRegion) => void;
  /** Active toepad slot label currently zoomed in the big image */
  activeSlotLabel?: string | null;
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
  isLegendOpen: boolean;
  activeModalSlotLabel: string | null;
}

const CLICK_THRESHOLD = 3;

export class ToepadPanels extends Component<ToepadPanelsProps, ToepadPanelsState> {
  private wasDragging = false;

  constructor(props: ToepadPanelsProps) {
    super(props);
    this.state = {
      drag: null,
      isLegendOpen: Boolean(props.defaultLegendOpen),
      activeModalSlotLabel: props.defaultActiveModalSlotLabel ?? null,
    };
  }

  public readonly handleToepadClick = (region: ToepadRegion): void => {
    if (!region.crop) return;
    if (this.props.onZoomToToe) {
      this.props.onZoomToToe(region);
    } else {
      this.openModal(region.slot.label);
    }
  };

  public readonly openModal = (slotLabel: string): void => {
    this.setState({ activeModalSlotLabel: slotLabel });
  };

  public readonly closeModal = (): void => {
    this.setState({ activeModalSlotLabel: null });
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
      this.wasDragging = true;
      setTimeout(() => {
        this.wasDragging = false;
      }, 100);
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

    const isActive = this.props.activeSlotLabel === slot.label;

    return (
      <div
        key={slot.label}
        data-testid={`toepad-card-${slot.label}`}
        style={{
          ...styles.card,
          ...(isActive ? styles.cardActive : {}),
        }}
        onClick={() => {
          if (crop && !this.wasDragging) {
            this.handleToepadClick(region);
          }
        }}
      >
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle} title={slot.label}>
            <span style={{ ...styles.swatch, backgroundColor: slot.color }} />
            {slot.title}
          </span>
          <div style={styles.cardHeaderActions}>
            {crop && <span style={styles.countChip}>{points.length} pts</span>}
            {crop && (
              <button
                type="button"
                data-testid={`toepad-zoom-button-${slot.label}`}
                style={{
                  ...styles.zoomButton,
                  ...(isActive ? styles.zoomButtonActive : {}),
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  this.handleToepadClick(region);
                }}
                title={isActive ? `Reset Zoom (${slot.title})` : `Zoom in on ${slot.title}`}
                aria-label={isActive ? `Reset Zoom (${slot.title})` : `Zoom in on ${slot.title}`}
              >
                {isActive ? "↺ Reset Zoom" : "⤢ Zoom to Toe"}
              </button>
            )}
          </div>
        </div>
        <div
          style={{
            ...styles.viewport,
            ...(isEditMode && crop ? styles.viewportEditing : {}),
            cursor: crop ? (isEditMode ? "crosshair" : "pointer") : "default",
          }}
          onClick={(e) => {
            if (crop && !this.wasDragging) {
              e.stopPropagation();
              this.handleToepadClick(region);
            }
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
                      onClick={(e) => {
                        e.stopPropagation();
                        if (this.props.selectedPoint?.id !== p.id) {
                          this.props.onPointSelect(p);
                        }
                      }}
                      style={{ cursor: isEditMode ? (isDragged ? "grabbing" : "grab") : "default" }}
                    >
                      {isEditMode && <circle cx={x} cy={y} r={r * 3} fill="transparent" />}
                      <circle cx={x} cy={y} r={r} fill={isSelected ? "yellow" : "red"} />
                      <text
                        x={x + r * 1.5}
                        y={y - r * 1.5}
                        fontSize={crop.size * 0.036}
                        fill="#ffffff"
                        stroke="rgba(0,0,0,0.85)"
                        strokeWidth={crop.size * 0.005}
                        paintOrder="stroke fill"
                        filter={`url(#toepad-label-shadow-${slot.label})`}
                        style={{ pointerEvents: "none", userSelect: "none", fontWeight: 700 }}
                      >
                        {p.id !== undefined ? p.id + 1 : i + 1}
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

  private readonly toggleLegend = (): void => {
    this.setState((prev) => ({ isLegendOpen: !prev.isLegendOpen }));
  };

  private renderControlsLegend(styles: ReturnType<typeof getToepadPanelsStyles>) {
    const { isLegendOpen } = this.state;
    return (
      <div style={styles.legendContainer}>
        <button
          onClick={this.toggleLegend}
          style={styles.legendHeaderButton}
          title="Toggle controls & shortcuts legend"
          aria-expanded={isLegendOpen}
        >
          <span style={styles.legendTitle}>
            <span>⌨️ Controls & Shortcuts Key</span>
          </span>
          <span style={styles.legendSubtitle}>
            {isLegendOpen ? "Hide ▲" : "Show ▼"}
          </span>
        </button>

        {isLegendOpen && (
          <div>
            {/* Table 2: Mouse & Touch Controls */}
            <div style={styles.legendSection}>
              <div style={styles.legendSectionTitle}>
                <span>🖱️ Mouse & Touch Controls</span>
              </div>
              <table style={styles.legendTable}>
                <thead>
                  <tr>
                    <th style={styles.legendTh}>Action</th>
                    <th style={styles.legendTh}>Mode</th>
                    <th style={styles.legendTh}>Effect</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={styles.legendTd}>Left-Click + Drag (bg)</td>
                    <td style={styles.legendTd}><span style={styles.modeBadgeView}>View</span></td>
                    <td style={styles.legendTd}>Pan around image</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}>Scroll Wheel</td>
                    <td style={styles.legendTd}><span style={styles.modeBadgeView}>View</span></td>
                    <td style={styles.legendTd}>Zoom in / out</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}>Left-Click (landmark)</td>
                    <td style={styles.legendTd}><span style={styles.modeBadgeAny}>Any</span></td>
                    <td style={styles.legendTd}>Select landmark (yellow)</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}>Left-Click + Drag (pt)</td>
                    <td style={styles.legendTd}><span style={styles.modeBadgeEdit}>Edit</span></td>
                    <td style={styles.legendTd}>Reposition landmark (synced)</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}>Right-Click</td>
                    <td style={styles.legendTd}><span style={styles.modeBadgeAny}>Any</span></td>
                    <td style={styles.legendTd}>Toggle Edit Mode</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Table 3: Keyboard Shortcuts */}
            <div style={styles.legendSection}>
              <div style={styles.legendSectionTitle}>
                <span>⌨️ Keyboard Shortcuts</span>
              </div>
              <table style={styles.legendTable}>
                <thead>
                  <tr>
                    <th style={styles.legendTh}>Key</th>
                    <th style={styles.legendTh}>Shortcut</th>
                    <th style={styles.legendTh}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={styles.legendTd}><kbd style={styles.kbd}>Ctrl+Z</kbd></td>
                    <td style={styles.legendTd}>Undo</td>
                    <td style={styles.legendTd}>Revert last landmark move</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}><kbd style={styles.kbd}>Right-Click</kbd></td>
                    <td style={styles.legendTd}>Edit Mode</td>
                    <td style={styles.legendTd}>Toggle edit / view mode</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}><kbd style={styles.kbd}>B</kbd></td>
                    <td style={styles.legendTd}>Boxes</td>
                    <td style={styles.legendTd}>Toggle bounding boxes</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}><kbd style={styles.kbd}>N</kbd></td>
                    <td style={styles.legendTd}>Labels</td>
                    <td style={styles.legendTd}>Toggle landmark labels (1–8)</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}><kbd style={styles.kbd}>O</kbd></td>
                    <td style={styles.legendTd}>Outline</td>
                    <td style={styles.legendTd}>Toggle solid / hollow circles</td>
                  </tr>
                  <tr>
                    <td style={styles.legendTd}><kbd style={styles.kbd}>T</kbd></td>
                    <td style={styles.legendTd}>Transparency</td>
                    <td style={styles.legendTd}>Cycle overlay transparency</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  }

  render() {
    const {
      imageURL,
      imageWidth,
      imageHeight,
      points,
      boundingBoxes,
      selectedPoint,
      isEditMode,
      theme,
    } = this.props;
    if (!imageURL || !imageWidth || !imageHeight) return null;

    const regions = getToepadRegions(points, boundingBoxes);
    const detected = regions.filter((r) => r.crop).length;
    const styles = getToepadPanelsStyles(theme);

    const activeRegion = this.state.activeModalSlotLabel
      ? regions.find(
          (r) => r.slot.label === this.state.activeModalSlotLabel && r.crop !== null
        )
      : null;

    return (
      <div style={{ ...styles.container, ...this.props.style }}>
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
        {this.renderControlsLegend(styles)}
        {activeRegion && !this.props.onZoomToToe && (
          <ToepadModal
            region={activeRegion}
            allPoints={points}
            imageURL={imageURL}
            imageWidth={imageWidth}
            imageHeight={imageHeight}
            selectedPoint={selectedPoint}
            isEditMode={isEditMode}
            onToggleEditMode={this.props.onToggleEditMode}
            onPointSelect={this.props.onPointSelect}
            onPointsChange={this.props.onPointsChange}
            onClose={this.closeModal}
            theme={theme}
          />
        )}
      </div>
    );
  }
}
