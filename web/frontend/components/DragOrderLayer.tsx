"use client";

import { PROVINCE_COORDINATES, UNIT_COLORS, getUnitCoordinates } from "@/lib/map-data";
import { getBaseProvince } from "@/lib/order-utils";
import type { DragState } from "@/hooks/useDragOrderState";

interface DragOrderLayerProps {
  dragState: DragState;
  humanPower: string;
  allUnits: Array<{ unit: string; power: string }>;
}

/**
 * SVG overlay layer for drag-and-drop order visualization.
 * Renders province highlights and drag preview arrows.
 */
export function DragOrderLayer({ dragState, humanPower, allUnits }: DragOrderLayerProps) {
  const humanColor = UNIT_COLORS[humanPower] || "#00bfff";

  if (dragState.phase === "idle") {
    return null;
  }

  // Get all valid destinations based on current phase
  let validDestinations: string[] = [];
  let highlightColor = "#22c55e"; // Green for move destinations
  let arrowStyle: "solid" | "dashed" = "solid";
  let arrowColor = humanColor;

  if (dragState.phase === "dragging") {
    // Combine all possible targets, but deduplicate
    const destSet = new Set<string>();
    dragState.validMoveDestinations.forEach((d) => destSet.add(getBaseProvince(d)));
    dragState.validSupportLocations.forEach((d) => destSet.add(getBaseProvince(d)));
    dragState.validConvoyArmyLocations.forEach((d) => destSet.add(getBaseProvince(d)));
    validDestinations = Array.from(destSet);
  } else if (dragState.phase === "selecting_support_target") {
    validDestinations = dragState.validTargetLocations;
    highlightColor = "#06b6d4"; // Cyan for support targets
  } else if (dragState.phase === "selecting_support_dest") {
    validDestinations = dragState.validDestinations;
    highlightColor = "#22c55e"; // Green
    arrowStyle = "dashed";
    arrowColor = "#22c55e";
  } else if (dragState.phase === "selecting_convoy_army") {
    validDestinations = dragState.validArmyLocations;
    highlightColor = "#a855f7"; // Purple for convoy
  } else if (dragState.phase === "selecting_convoy_dest") {
    validDestinations = dragState.validDestinations;
    highlightColor = "#a855f7"; // Purple
    arrowStyle = "dashed";
    arrowColor = "#a855f7";
  }

  // Get unit coordinates for arrow start
  const unitCoords = getUnitCoordinates(dragState.unit);
  if (!unitCoords) return null;

  // For multi-step phases, show arrow from unit to target
  let arrowStartCoords = unitCoords;
  if (dragState.phase === "selecting_support_dest") {
    const targetCoords = PROVINCE_COORDINATES[dragState.targetLocation];
    if (targetCoords) {
      arrowStartCoords = targetCoords;
    }
  } else if (dragState.phase === "selecting_convoy_dest") {
    const armyCoords = PROVINCE_COORDINATES[dragState.armyLocation];
    if (armyCoords) {
      arrowStartCoords = armyCoords;
    }
  }

  return (
    <g id="drag-layer">
      {/* Province highlights */}
      {validDestinations.map((dest, idx) => {
        const baseDest = getBaseProvince(dest);
        const coords = PROVINCE_COORDINATES[baseDest] || PROVINCE_COORDINATES[dest];
        if (!coords) return null;

        // Check if this is a unit location (for support)
        const isUnitLocation = allUnits.some((u) => {
          const loc = u.unit.split(" ")[1];
          return getBaseProvince(loc) === baseDest;
        });

        // Different highlight for move destinations vs support targets
        let highlightStyle = highlightColor;
        if (dragState.phase === "dragging") {
          if (dragState.validSupportLocations.some((s) => getBaseProvince(s) === baseDest)) {
            highlightStyle = "#06b6d4"; // Cyan for supportable units
          } else if (dragState.validConvoyArmyLocations.some((a) => getBaseProvince(a) === baseDest)) {
            highlightStyle = "#a855f7"; // Purple for convoyable armies
          }
        }

        return (
          <g key={`highlight-${baseDest}-${idx}`}>
            {/* Outer glow */}
            <circle
              cx={coords.x}
              cy={coords.y}
              r="40"
              fill={highlightStyle}
              opacity="0.15"
              className="animate-pulse"
            />
            {/* Inner highlight */}
            <circle
              cx={coords.x}
              cy={coords.y}
              r="30"
              fill="none"
              stroke={highlightStyle}
              strokeWidth="3"
              opacity="0.7"
              className="animate-pulse"
            />
            {/* Unit indicator for support targets */}
            {isUnitLocation && dragState.phase === "dragging" && (
              <circle
                cx={coords.x}
                cy={coords.y}
                r="20"
                fill={highlightStyle}
                opacity="0.3"
              />
            )}
          </g>
        );
      })}

      {/* Drag preview arrow (only during active drag) */}
      {dragState.phase === "dragging" && (
        <DragPreviewArrow
          startX={unitCoords.x}
          startY={unitCoords.y}
          endX={dragState.currentPos.x}
          endY={dragState.currentPos.y}
          color={humanColor}
          style="solid"
        />
      )}

      {/* Selected unit highlight */}
      <circle
        cx={unitCoords.x}
        cy={unitCoords.y}
        r="35"
        fill="none"
        stroke="#fbbf24"
        strokeWidth="4"
        className="animate-pulse"
      />

      {/* Phase indicator for multi-step operations */}
      {(dragState.phase === "selecting_support_target" ||
        dragState.phase === "selecting_support_dest" ||
        dragState.phase === "selecting_convoy_army" ||
        dragState.phase === "selecting_convoy_dest") && (
        <PhaseIndicator
          phase={dragState.phase}
          x={unitCoords.x}
          y={unitCoords.y - 60}
        />
      )}

      {/* Show intermediate connections for multi-step */}
      {dragState.phase === "selecting_support_dest" && (
        <IntermediateConnection
          fromCoords={unitCoords}
          toLocation={dragState.targetLocation}
          color="#06b6d4"
        />
      )}
      {dragState.phase === "selecting_convoy_dest" && (
        <IntermediateConnection
          fromCoords={unitCoords}
          toLocation={dragState.armyLocation}
          color="#a855f7"
        />
      )}
    </g>
  );
}

/**
 * Arrow preview during drag.
 */
function DragPreviewArrow({
  startX,
  startY,
  endX,
  endY,
  color,
  style,
}: {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  color: string;
  style: "solid" | "dashed";
}) {
  // Calculate arrow head direction
  const angle = Math.atan2(endY - startY, endX - startX);
  const headLength = 15;
  const headAngle = Math.PI / 6;

  const headX1 = endX - headLength * Math.cos(angle - headAngle);
  const headY1 = endY - headLength * Math.sin(angle - headAngle);
  const headX2 = endX - headLength * Math.cos(angle + headAngle);
  const headY2 = endY - headLength * Math.sin(angle + headAngle);

  return (
    <g>
      {/* Shadow */}
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke="black"
        strokeWidth="10"
        opacity="0.3"
        strokeLinecap="round"
      />
      {/* Arrow line */}
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke={color}
        strokeWidth="6"
        strokeDasharray={style === "dashed" ? "15,8" : "none"}
        strokeLinecap="round"
      />
      {/* Arrow head */}
      <polygon
        points={`${endX},${endY} ${headX1},${headY1} ${headX2},${headY2}`}
        fill={color}
      />
    </g>
  );
}

/**
 * Phase indicator badge for multi-step operations.
 */
function PhaseIndicator({
  phase,
  x,
  y,
}: {
  phase: string;
  x: number;
  y: number;
}) {
  let text = "";
  let bgColor = "#374151";

  switch (phase) {
    case "selecting_support_target":
      text = "Select unit to support";
      bgColor = "#0891b2"; // Cyan
      break;
    case "selecting_support_dest":
      text = "Select destination";
      bgColor = "#16a34a"; // Green
      break;
    case "selecting_convoy_army":
      text = "Select army to convoy";
      bgColor = "#9333ea"; // Purple
      break;
    case "selecting_convoy_dest":
      text = "Select convoy destination";
      bgColor = "#9333ea"; // Purple
      break;
  }

  const textWidth = text.length * 7;
  const padding = 12;
  const height = 28;

  return (
    <g transform={`translate(${x - textWidth / 2 - padding}, ${y})`}>
      {/* Background */}
      <rect
        x="0"
        y="0"
        width={textWidth + padding * 2}
        height={height}
        rx="6"
        fill={bgColor}
        opacity="0.95"
      />
      {/* Text */}
      <text
        x={textWidth / 2 + padding}
        y={height / 2 + 5}
        textAnchor="middle"
        fontSize="12"
        fontWeight="600"
        fill="white"
      >
        {text}
      </text>
    </g>
  );
}

/**
 * Line showing connection in multi-step operations.
 */
function IntermediateConnection({
  fromCoords,
  toLocation,
  color,
}: {
  fromCoords: { x: number; y: number };
  toLocation: string;
  color: string;
}) {
  const toCoords = PROVINCE_COORDINATES[toLocation];
  if (!toCoords) return null;

  return (
    <g>
      <line
        x1={fromCoords.x}
        y1={fromCoords.y}
        x2={toCoords.x}
        y2={toCoords.y}
        stroke={color}
        strokeWidth="4"
        strokeDasharray="8,4"
        opacity="0.6"
      />
      <circle
        cx={toCoords.x}
        cy={toCoords.y}
        r="25"
        fill={color}
        opacity="0.3"
      />
    </g>
  );
}
