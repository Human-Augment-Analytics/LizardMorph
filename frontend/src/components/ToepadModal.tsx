import { Component } from "react";
import type React from "react";
import type { Point } from "../models/Point";
import type { ResolvedTheme } from "../contexts/theme";
import type { ToepadRegion } from "../services/ToepadRegions";
import { getToepadModalStyles } from "./ToepadModal.style";

export interface ToepadModalProps {
  region: ToepadRegion;
  /** All specimen landmarks in image-pixel space. */
  allPoints: Point[];
  imageURL: string | null;
  imageWidth: number;
  imageHeight: number;
  selectedPoint: Point | null;
  isEditMode: boolean;
  onToggleEditMode: () => void;
  onPointSelect: (point: Point | null) => void;
  onPointsChange: (points: Point[]) => void;
  onClose: () => void;
  theme: ResolvedTheme;
  initialZoom?: number;
  initialPan?: { x: number; y: number };
}

interface DragState {
  id: number;
  x: number;
  y: number;
  startClientX: number;
  startClientY: number;
  moved: boolean;
}

interface PanState {
  clientX: number;
  clientY: number;
  startPanX: number;
  startPanY: number;
}

export interface ToepadModalState {
  zoom: number;
  pan: { x: number; y: number };
  drag: DragState | null;
  isPanning: boolean;
  undoStack: Point[][];
}

const CLICK_THRESHOLD = 3;
const MAX_UNDO_STACK = 30;

export class ToepadModal extends Component<ToepadModalProps, ToepadModalState> {
  private panState: PanState | null = null;
  private viewportRef: HTMLDivElement | null = null;
  private dialogRef: HTMLDivElement | null = null;
  private previousActiveElement: HTMLElement | null = null;

  constructor(props: ToepadModalProps) {
    super(props);
    this.state = {
      zoom: props.initialZoom ?? 1,
      pan: props.initialPan ?? { x: 0, y: 0 },
      drag: null,
      isPanning: false,
      undoStack: [],
    };
  }

  componentDidMount(): void {
    if (typeof document !== "undefined") {
      this.previousActiveElement = document.activeElement as HTMLElement | null;
    }
    if (this.dialogRef) {
      this.dialogRef.focus();
    }
    if (typeof window !== "undefined") {
      window.addEventListener("keydown", this.handleKeyDown);
    }
    if (this.viewportRef) {
      this.viewportRef.addEventListener("wheel", this.handleNativeWheel, { passive: false });
    }
  }

  componentWillUnmount(): void {
    if (typeof window !== "undefined") {
      window.removeEventListener("keydown", this.handleKeyDown);
    }
    if (this.viewportRef) {
      this.viewportRef.removeEventListener("wheel", this.handleNativeWheel);
    }
    if (this.previousActiveElement && typeof this.previousActiveElement.focus === "function") {
      this.previousActiveElement.focus();
    }
  }

  public readonly handleKeyDown = (event: KeyboardEvent): void => {
    if (event.key === "Escape") {
      event.preventDefault();
      this.props.onClose();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      this.handleUndo();
      return;
    }
  };

  public readonly handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>): void => {
    if (event.target === event.currentTarget) {
      this.props.onClose();
    }
  };

  public readonly handleZoomIn = (): void => {
    this.setState((prev) => ({
      zoom: Math.min(Math.round(prev.zoom * 1.25 * 100) / 100, 10),
    }));
  };

  public readonly handleZoomOut = (): void => {
    this.setState((prev) => ({
      zoom: Math.max(Math.round((prev.zoom / 1.25) * 100) / 100, 0.5),
    }));
  };

  public readonly handleResetZoom = (): void => {
    this.setState({
      zoom: 1,
      pan: { x: 0, y: 0 },
    });
  };

  public readonly handleUndo = (): void => {
    const { undoStack } = this.state;
    if (undoStack.length === 0) return;
    const previousPoints = undoStack[undoStack.length - 1];
    this.setState({
      undoStack: undoStack.slice(0, -1),
    });
    this.props.onPointsChange(previousPoints);
  };

  public readonly toImagePoint = (
    svg: SVGSVGElement,
    clientX: number,
    clientY: number
  ): { x: number; y: number } | null => {
    const ctm = svg.getScreenCTM?.();
    if (ctm) {
      const pt = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
      return {
        x: Math.min(Math.max(pt.x, 0), this.props.imageWidth),
        y: Math.min(Math.max(pt.y, 0), this.props.imageHeight),
      };
    }
    const rect = svg.getBoundingClientRect?.();
    if (!rect || !rect.width || !rect.height) return null;
    const { crop } = this.props.region;
    if (!crop) return null;
    const normX = (clientX - rect.left) / rect.width;
    const normY = (clientY - rect.top) / rect.height;
    return {
      x: Math.min(Math.max(crop.x + normX * crop.size, 0), this.props.imageWidth),
      y: Math.min(Math.max(crop.y + normY * crop.size, 0), this.props.imageHeight),
    };
  };

  public readonly handlePointPointerDown = (
    event: React.PointerEvent<SVGGElement>,
    point: Point
  ): void => {
    if (!this.props.isEditMode || event.button !== 0) return;
    if (event.shiftKey || event.altKey) {
      // When Shift or Alt is pressed, do not capture so event bubbles to canvas panning
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.ownerSVGElement?.setPointerCapture?.(event.pointerId);

    if (this.props.selectedPoint?.id !== point.id) {
      this.props.onPointSelect(point);
    }

    this.setState((prev) => ({
      drag: {
        id: point.id,
        x: point.x,
        y: point.y,
        startClientX: event.clientX,
        startClientY: event.clientY,
        moved: false,
      },
      undoStack: [...prev.undoStack.slice(-MAX_UNDO_STACK + 1), this.props.allPoints],
    }));
  };

  public readonly handleSvgPointerDown = (
    event: React.PointerEvent<SVGSVGElement>
  ): void => {
    if (event.button !== 0) return;
    const canPan = !this.props.isEditMode || event.shiftKey || event.altKey;
    if (canPan) {
      event.preventDefault();
      event.currentTarget.setPointerCapture?.(event.pointerId);
      this.panState = {
        clientX: event.clientX,
        clientY: event.clientY,
        startPanX: this.state.pan.x,
        startPanY: this.state.pan.y,
      };
      this.setState({ isPanning: true });
    } else if (this.props.isEditMode && this.props.selectedPoint) {
      this.props.onPointSelect(null);
    }
  };

  public readonly handlePointerMove = (
    event: React.PointerEvent<SVGSVGElement>
  ): void => {
    const { drag } = this.state;
    if (drag) {
      const moved =
        drag.moved ||
        Math.hypot(
          event.clientX - drag.startClientX,
          event.clientY - drag.startClientY
        ) >= CLICK_THRESHOLD;
      if (!moved) return;
      const pt = this.toImagePoint(
        event.currentTarget,
        event.clientX,
        event.clientY
      );
      if (!pt) return;
      this.setState({ drag: { ...drag, ...pt, moved: true } });
      return;
    }
    if (this.panState) {
      const dx = event.clientX - this.panState.clientX;
      const dy = event.clientY - this.panState.clientY;
      this.setState({
        pan: {
          x: this.panState.startPanX + dx,
          y: this.panState.startPanY + dy,
        },
      });
    }
  };

  public readonly handlePointerUp = (
    event: React.PointerEvent<SVGSVGElement>
  ): void => {
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture?.(event.pointerId);
    }
    const { drag } = this.state;
    if (drag) {
      if (drag.moved) {
        const updated = this.props.allPoints.map((p) =>
          p.id === drag.id ? { ...p, x: drag.x, y: drag.y } : p
        );
        this.props.onPointsChange(updated);
      } else {
        // Did not move: discard the unused undo snapshot
        this.setState((prev) => ({
          undoStack: prev.undoStack.slice(0, -1),
        }));
      }
      this.setState({ drag: null });
    }
    if (this.panState) {
      this.panState = null;
      this.setState({ isPanning: false });
    }
  };

  public readonly handlePointerCancel = (): void => {
    const { drag } = this.state;
    if (drag) {
      this.setState((prev) => ({
        drag: null,
        undoStack: prev.undoStack.slice(0, -1),
      }));
    }
    if (this.panState) {
      this.panState = null;
      this.setState({ isPanning: false });
    }
  };

  private readonly handleNativeWheel = (event: WheelEvent): void => {
    event.preventDefault();
    event.stopPropagation();
    if (event.deltaY === 0) return;
    const factor = event.deltaY < 0 ? 1.08 : 0.92;
    this.setState((prev) => ({
      zoom: Math.min(Math.max(Math.round(prev.zoom * factor * 100) / 100, 0.5), 10),
    }));
  };

  public readonly handleWheel = (event: React.WheelEvent<HTMLDivElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    if (event.deltaY === 0) return;
    const factor = event.deltaY < 0 ? 1.08 : 0.92;
    this.setState((prev) => ({
      zoom: Math.min(Math.max(Math.round(prev.zoom * factor * 100) / 100, 0.5), 10),
    }));
  };

  public readonly handleContextMenu = (event: React.MouseEvent): void => {
    event.preventDefault();
    this.props.onToggleEditMode();
  };

  render() {
    const {
      region,
      imageURL,
      imageWidth,
      imageHeight,
      selectedPoint,
      isEditMode,
      theme,
      onToggleEditMode,
      onClose,
    } = this.props;
    const { zoom, pan, drag, isPanning } = this.state;
    const styles = getToepadModalStyles(theme, isEditMode);
    const { slot, crop, points } = region;

    return (
      <div
        style={styles.backdrop}
        onClick={this.handleBackdropClick}
        data-testid="toepad-modal-backdrop"
      >
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="toepad-modal-title"
          tabIndex={-1}
          ref={(el) => {
            this.dialogRef = el;
          }}
          style={{ ...styles.dialog, outline: "none" }}
          onClick={(e) => e.stopPropagation()}
          data-testid="toepad-modal-dialog"
        >
          {/* Header */}
          <div style={styles.header}>
            <div style={styles.headerLeft}>
              <h2 id="toepad-modal-title" style={styles.title}>
                <span style={{ ...styles.swatch, backgroundColor: slot.color }} />
                <span>{slot.title} Close-up</span>
              </h2>
              {crop && <span style={styles.countChip}>{points.length} pts</span>}
            </div>

            <div style={styles.headerControls}>
              <button
                type="button"
                onClick={onToggleEditMode}
                style={styles.editButton}
                title="Toggle Edit Mode (Right-click)"
                aria-pressed={isEditMode}
              >
                ✏️ Edit Points: {isEditMode ? "On" : "Off"}
              </button>

              <button
                type="button"
                onClick={this.handleUndo}
                disabled={this.state.undoStack.length === 0}
                style={{
                  ...styles.undoButton,
                  opacity: this.state.undoStack.length === 0 ? 0.45 : 1,
                  cursor: this.state.undoStack.length === 0 ? "not-allowed" : "pointer",
                }}
                title="Undo landmark move (Ctrl+Z)"
                aria-label="Undo landmark move"
              >
                ↶ Undo
              </button>

              <div style={styles.zoomButtonGroup}>
                <button
                  type="button"
                  onClick={this.handleZoomIn}
                  disabled={zoom >= 10}
                  style={{
                    ...styles.zoomButton,
                    opacity: zoom >= 10 ? 0.45 : 1,
                    cursor: zoom >= 10 ? "not-allowed" : "pointer",
                  }}
                  title="Zoom In"
                  aria-label="Zoom In"
                >
                  + Zoom In
                </button>
                <button
                  type="button"
                  onClick={this.handleZoomOut}
                  disabled={zoom <= 0.5}
                  style={{
                    ...styles.zoomButton,
                    opacity: zoom <= 0.5 ? 0.45 : 1,
                    cursor: zoom <= 0.5 ? "not-allowed" : "pointer",
                  }}
                  title="Zoom Out"
                  aria-label="Zoom Out"
                >
                  - Zoom Out
                </button>
                <span style={styles.zoomBadge}>{Math.round(zoom * 100)}%</span>
                <button
                  type="button"
                  onClick={this.handleResetZoom}
                  style={styles.zoomButtonReset}
                  title="Reset Zoom"
                  aria-label="Reset Zoom"
                >
                  ↺ Reset
                </button>
              </div>

              <button
                type="button"
                onClick={onClose}
                style={styles.closeButton}
                title="Close Modal (Escape)"
                aria-label="Close modal"
              >
                ✕ Close
              </button>
            </div>
          </div>

          {/* Viewport */}
          <div
            ref={(el) => {
              this.viewportRef = el;
            }}
            style={styles.viewportContainer}
            onWheel={this.handleWheel}
            data-testid="toepad-modal-viewport"
          >
            {!crop || !imageURL ? (
              <div style={styles.emptyState}>Not detected</div>
            ) : (
              <div
                style={{
                  ...styles.viewportContent,
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: "center center",
                }}
              >
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
                  style={{
                    ...styles.overlay,
                    cursor: isEditMode
                      ? drag?.moved
                        ? "grabbing"
                        : isPanning
                        ? "grabbing"
                        : "crosshair"
                      : isPanning
                      ? "grabbing"
                      : "grab",
                  }}
                  onPointerDown={this.handleSvgPointerDown}
                  onPointerMove={this.handlePointerMove}
                  onPointerUp={this.handlePointerUp}
                  onPointerCancel={this.handlePointerCancel}
                  onContextMenu={this.handleContextMenu}
                >
                  <defs>
                    <filter
                      id={`toepad-modal-shadow-${slot.label}`}
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
                        data-testid={`modal-landmark-${p.id}`}
                        onPointerDown={(e) => this.handlePointPointerDown(e, p)}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (selectedPoint?.id !== p.id) {
                            this.props.onPointSelect(p);
                          }
                        }}
                        style={{
                          cursor: isEditMode
                            ? isDragged
                              ? "grabbing"
                              : "grab"
                            : "default",
                        }}
                      >
                        {isEditMode && (
                          <circle cx={x} cy={y} r={r * 3.5} fill="transparent" />
                        )}
                        <circle
                          cx={x}
                          cy={y}
                          r={r + crop.size * 0.002}
                          fill="rgba(0,0,0,0.6)"
                        />
                        <circle
                          cx={x}
                          cy={y}
                          r={r}
                          fill={isSelected ? "yellow" : "red"}
                        />
                        <text
                          x={x + r * 1.5}
                          y={y - r * 1.5}
                          fontSize={crop.size * 0.036}
                          fill="#ffffff"
                          stroke="rgba(0,0,0,0.85)"
                          strokeWidth={crop.size * 0.005}
                          paintOrder="stroke fill"
                          filter={`url(#toepad-modal-shadow-${slot.label})`}
                          style={{
                            pointerEvents: "none",
                            userSelect: "none",
                            fontWeight: 700,
                          }}
                        >
                          {p.id !== undefined ? p.id + 1 : i + 1}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            )}
          </div>

          {/* Footer info/shortcut bar */}
          <div style={styles.footer}>
            <span>
              {isEditMode
                ? "✏️ Drag landmarks to reposition. Shift / Alt + Drag to Pan in Edit Mode."
                : "🖱️ Drag to pan • Scroll to zoom. Click 'Edit Points' to adjust."}
            </span>
            <span>
              <kbd style={styles.kbd}>Ctrl+Z</kbd> Undo &nbsp;•&nbsp;{" "}
              <kbd style={styles.kbd}>Right-Click</kbd> Toggle Edit &nbsp;•&nbsp;{" "}
              <kbd style={styles.kbd}>Esc</kbd> Close
            </span>
          </div>
        </div>
      </div>
    );
  }
}
