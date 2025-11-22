import React, { Component } from "react";
import { MeasurementsPanelStyles } from "./MeasurementsPanel.style";
import type { Measurement } from "../models/Measurement";
import type { Point } from "../models/Point";
import type { ScaleSettings } from "../models/ScaleSettings";

interface MeasurementsPanelProps {
  points: Point[];
  measurements: Measurement[];
  scaleSettings: ScaleSettings;
  onMeasurementsChange: (measurements: Measurement[]) => void;
}

export class MeasurementsPanel extends Component<MeasurementsPanelProps> {
  componentDidUpdate(prevProps: MeasurementsPanelProps): void {
    // Recalculate distances when scale settings or points change
    if (
      prevProps.scaleSettings !== this.props.scaleSettings ||
      prevProps.points !== this.props.points
    ) {
      const updatedMeasurements = this.props.measurements.map((measurement) => {
        const pointA = measurement.pointAId
          ? this.props.points.find((p) => p.id === measurement.pointAId)
          : null;
        const pointB = measurement.pointBId
          ? this.props.points.find((p) => p.id === measurement.pointBId)
          : null;

        return {
          ...measurement,
          calculatedDistance: this.calculateDistance(
            pointA,
            pointB,
            this.props.scaleSettings
          ),
        };
      });

      // Only update if measurements actually changed
      const hasChanges = updatedMeasurements.some(
        (m, i) =>
          m.calculatedDistance !== this.props.measurements[i]?.calculatedDistance
      );

      if (hasChanges) {
        this.props.onMeasurementsChange(updatedMeasurements);
      }
    }
  }

  private calculateDistance(
    pointA: Point | null,
    pointB: Point | null,
    scaleSettings: ScaleSettings
  ): number | null {
    if (!pointA || !pointB || !scaleSettings.pointAId || !scaleSettings.pointBId || scaleSettings.value === null || scaleSettings.value <= 0) {
      return null;
    }

    // Calculate pixel distance between the two measurement points
    const pixelDistance = Math.sqrt(
      Math.pow(pointB.x - pointA.x, 2) + Math.pow(pointB.y - pointA.y, 2)
    );

    // Find the scale points
    const scalePointA = this.props.points.find(
      (p) => p.id === scaleSettings.pointAId
    );
    const scalePointB = this.props.points.find(
      (p) => p.id === scaleSettings.pointBId
    );

    if (!scalePointA || !scalePointB) {
      return null;
    }

    // Calculate pixel distance of the scale
    const scalePixelDistance = Math.sqrt(
      Math.pow(scalePointB.x - scalePointA.x, 2) +
        Math.pow(scalePointB.y - scalePointA.y, 2)
    );

    if (scalePixelDistance === 0) {
      return null;
    }

    // Calculate the real-world distance
    // pixelDistance / scalePixelDistance = realDistance / scaleValue
    // realDistance = (pixelDistance / scalePixelDistance) * scaleValue
    const realDistance = (pixelDistance / scalePixelDistance) * scaleSettings.value;

    return realDistance;
  }

  private handleAddMeasurement = (): void => {
    const newMeasurement: Measurement = {
      id: `measurement-${Date.now()}`,
      label: "",
      pointAId: null,
      pointBId: null,
      calculatedDistance: null,
    };

    this.props.onMeasurementsChange([
      ...this.props.measurements,
      newMeasurement,
    ]);
  };

  private handleDeleteMeasurement = (id: string): void => {
    this.props.onMeasurementsChange(
      this.props.measurements.filter((m) => m.id !== id)
    );
  };

  private handleMeasurementChange = (
    id: string,
    updates: Partial<Measurement>
  ): void => {
    const updatedMeasurements = this.props.measurements.map((m) => {
      if (m.id === id) {
        const updated = { ...m, ...updates };
        
        // Recalculate distance if points changed
        if (updates.pointAId !== undefined || updates.pointBId !== undefined) {
          const pointA = updated.pointAId
            ? this.props.points.find((p) => p.id === updated.pointAId)
            : null;
          const pointB = updated.pointBId
            ? this.props.points.find((p) => p.id === updated.pointBId)
            : null;

          updated.calculatedDistance = this.calculateDistance(
            pointA,
            pointB,
            this.props.scaleSettings
          );
        }

        return updated;
      }
      return m;
    });

    this.props.onMeasurementsChange(updatedMeasurements);
  };

  render() {
    const { points, measurements, scaleSettings } = this.props;

    return (
      <div style={MeasurementsPanelStyles.container}>
        <h3 style={MeasurementsPanelStyles.title}>Measurements</h3>

        {measurements.length === 0 ? (
          <div style={MeasurementsPanelStyles.emptyState}>
            No measurements yet. Click "Add Measurement" to create one.
          </div>
        ) : (
          measurements.map((measurement, index) => {
            const pointA = measurement.pointAId
              ? points.find((p) => p.id === measurement.pointAId)
              : null;
            const pointB = measurement.pointBId
              ? points.find((p) => p.id === measurement.pointBId)
              : null;

            // Recalculate distance if scale settings changed
            let calculatedDistance = measurement.calculatedDistance;
            if (pointA && pointB) {
              calculatedDistance = this.calculateDistance(
                pointA,
                pointB,
                scaleSettings
              );
            }

            return (
              <div key={measurement.id} style={MeasurementsPanelStyles.measurementItem}>
                <div style={MeasurementsPanelStyles.measurementHeader}>
                  <span style={MeasurementsPanelStyles.measurementLabel}>
                    Measurement #{index + 1}
                  </span>
                  <button
                    style={MeasurementsPanelStyles.deleteButton}
                    onClick={() => this.handleDeleteMeasurement(measurement.id)}
                  >
                    Delete
                  </button>
                </div>

                <div style={MeasurementsPanelStyles.formGroup}>
                  <label style={MeasurementsPanelStyles.label}>Label:</label>
                  <input
                    type="text"
                    style={MeasurementsPanelStyles.input}
                    value={measurement.label}
                    onChange={(e) =>
                      this.handleMeasurementChange(measurement.id, {
                        label: e.target.value,
                      })
                    }
                    placeholder="e.g., Head width"
                  />
                </div>

                <div style={MeasurementsPanelStyles.formGroup}>
                  <label style={MeasurementsPanelStyles.label}>Point A:</label>
                  <select
                    style={MeasurementsPanelStyles.select}
                    value={measurement.pointAId ?? ""}
                    onChange={(e) =>
                      this.handleMeasurementChange(measurement.id, {
                        pointAId: e.target.value
                          ? parseInt(e.target.value, 10)
                          : null,
                      })
                    }
                  >
                    <option value="">Select landmark...</option>
                    {points.map((point) => (
                      <option key={point.id} value={point.id}>
                        Landmark {point.id}
                      </option>
                    ))}
                  </select>
                </div>

                <div style={MeasurementsPanelStyles.formGroup}>
                  <label style={MeasurementsPanelStyles.label}>Point B:</label>
                  <select
                    style={MeasurementsPanelStyles.select}
                    value={measurement.pointBId ?? ""}
                    onChange={(e) =>
                      this.handleMeasurementChange(measurement.id, {
                        pointBId: e.target.value
                          ? parseInt(e.target.value, 10)
                          : null,
                      })
                    }
                  >
                    <option value="">Select landmark...</option>
                    {points.map((point) => (
                      <option key={point.id} value={point.id}>
                        Landmark {point.id}
                      </option>
                    ))}
                  </select>
                </div>

                {calculatedDistance !== null && (
                  <div style={MeasurementsPanelStyles.distanceDisplay}>
                    Distance: {calculatedDistance.toFixed(3)} {scaleSettings.units}
                    {measurement.label && ` (${measurement.label})`}
                  </div>
                )}
              </div>
            );
          })
        )}

        <button
          style={MeasurementsPanelStyles.addButton}
          onClick={this.handleAddMeasurement}
        >
          + Add Measurement
        </button>
      </div>
    );
  }
}

