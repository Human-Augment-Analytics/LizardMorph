import React, { Component } from "react";
import { ScaleSettingsStyles } from "./ScaleSettings.style";
import type { ScaleSettings as ScaleSettingsType } from "../models/ScaleSettings";
import type { Point } from "../models/Point";
import { UNITS } from "../models/ScaleSettings";

interface ScaleSettingsProps {
  points: Point[];
  scaleSettings: ScaleSettingsType;
  onScaleSettingsChange: (settings: ScaleSettingsType) => void;
}

export class ScaleSettings extends Component<ScaleSettingsProps> {
  render() {
    const { points, scaleSettings, onScaleSettingsChange } = this.props;

    const handlePointAChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      const pointAId = e.target.value ? parseInt(e.target.value, 10) : null;
      onScaleSettingsChange({
        ...scaleSettings,
        pointAId,
      });
    };

    const handlePointBChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      const pointBId = e.target.value ? parseInt(e.target.value, 10) : null;
      onScaleSettingsChange({
        ...scaleSettings,
        pointBId,
      });
    };

    const handleValueChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value ? parseFloat(e.target.value) : null;
      onScaleSettingsChange({
        ...scaleSettings,
        value,
      });
    };

    const handleUnitsChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      onScaleSettingsChange({
        ...scaleSettings,
        units: e.target.value,
      });
    };

    return (
      <div style={ScaleSettingsStyles.container}>
        <h3 style={ScaleSettingsStyles.title}>Set Scale</h3>
        
        <div style={ScaleSettingsStyles.formGroup}>
          <label style={ScaleSettingsStyles.label}>Point A:</label>
          <select
            style={ScaleSettingsStyles.select}
            value={scaleSettings.pointAId ?? ""}
            onChange={handlePointAChange}
          >
            <option value="">Select landmark...</option>
            {points.map((point) => (
              <option key={point.id} value={point.id}>
                Landmark {point.id}
              </option>
            ))}
          </select>
        </div>

        <div style={ScaleSettingsStyles.formGroup}>
          <label style={ScaleSettingsStyles.label}>Point B:</label>
          <select
            style={ScaleSettingsStyles.select}
            value={scaleSettings.pointBId ?? ""}
            onChange={handlePointBChange}
          >
            <option value="">Select landmark...</option>
            {points.map((point) => (
              <option key={point.id} value={point.id}>
                Landmark {point.id}
              </option>
            ))}
          </select>
        </div>

        <div style={ScaleSettingsStyles.formGroup}>
          <label style={ScaleSettingsStyles.label}>Value:</label>
          <input
            type="number"
            style={ScaleSettingsStyles.input}
            value={scaleSettings.value ?? ""}
            onChange={handleValueChange}
            placeholder="Enter known distance"
            step="any"
            min="0"
          />
        </div>

        <div style={ScaleSettingsStyles.formGroup}>
          <label style={ScaleSettingsStyles.label}>Units:</label>
          <select
            style={ScaleSettingsStyles.select}
            value={scaleSettings.units}
            onChange={handleUnitsChange}
          >
            {UNITS.map((unit) => (
              <option key={unit.value} value={unit.value}>
                {unit.label}
              </option>
            ))}
          </select>
        </div>

        {scaleSettings.pointAId &&
          scaleSettings.pointBId &&
          scaleSettings.value !== null &&
          scaleSettings.value > 0 && (
            <div style={ScaleSettingsStyles.infoText}>
              Scale set: {scaleSettings.value} {scaleSettings.units} between
              Landmark {scaleSettings.pointAId} and Landmark{" "}
              {scaleSettings.pointBId}
            </div>
          )}
      </div>
    );
  }
}

