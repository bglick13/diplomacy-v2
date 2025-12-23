"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import {
  PROVINCE_COORDINATES,
  MAP_VIEWBOX,
  UNIT_COLORS,
  parseUnitLocation,
  isFleet,
  getUnitCoordinates,
  svgPointFromEvent,
  findNearestProvinceFromAll,
} from "@/lib/map-data";
import { DragOrderLayer } from "./DragOrderLayer";
import type { DragState, Point } from "@/hooks/useDragOrderState";

interface DiplomacyMapProps {
  units: Record<string, string[]>;
  centers: Record<string, string[]>;
  humanPower: string;
  validMoves?: Record<string, string[]>;
  selectedOrders?: Record<string, string>;
  selectedUnit?: string | null;
  onUnitClick?: (unit: string) => void;
  // Drag-and-drop props
  dragState?: DragState;
  onDragStart?: (unit: string, pos: Point) => void;
  onDragMove?: (pos: Point) => void;
  onDragEnd?: (province: string | null) => void;
  onDragCancel?: () => void;
}

// Order types for visualization
type OrderType = "hold" | "move" | "support" | "convoy" | "build" | "disband" | "retreat";

interface ParsedOrder {
  type: OrderType;
  unit: string;
  from: string;
  to?: string;
  supportTarget?: string;
  supportFrom?: string;
  supportTo?: string;
}

/**
 * Parse an order string into structured data for visualization
 */
function parseOrder(unit: string, order: string): ParsedOrder | null {
  const fromLoc = parseUnitLocation(unit);

  // Hold: "A PAR H"
  if (order.includes(" H")) {
    return { type: "hold", unit, from: fromLoc };
  }

  // Move: "A PAR - BUR" or "F BRE - MAO"
  if (order.includes(" - ")) {
    const match = order.match(/- ([A-Z]{3}(?:\/[A-Z]{2})?)/);
    if (match) {
      return { type: "move", unit, from: fromLoc, to: match[1] };
    }
  }

  // Support hold: "A PAR S A BUR"
  // Support move: "A PAR S A BUR - MAR"
  if (order.includes(" S ")) {
    const supportMoveMatch = order.match(/S [AF] ([A-Z]{3}(?:\/[A-Z]{2})?) - ([A-Z]{3}(?:\/[A-Z]{2})?)/);
    if (supportMoveMatch) {
      return {
        type: "support",
        unit,
        from: fromLoc,
        supportFrom: supportMoveMatch[1],
        supportTo: supportMoveMatch[2],
      };
    }
    const supportHoldMatch = order.match(/S [AF] ([A-Z]{3}(?:\/[A-Z]{2})?)/);
    if (supportHoldMatch) {
      return {
        type: "support",
        unit,
        from: fromLoc,
        supportTarget: supportHoldMatch[1],
      };
    }
  }

  // Convoy: "F MAO C A PAR - LON"
  if (order.includes(" C ")) {
    const convoyMatch = order.match(/C [AF] ([A-Z]{3}) - ([A-Z]{3})/);
    if (convoyMatch) {
      return {
        type: "convoy",
        unit,
        from: fromLoc,
        supportFrom: convoyMatch[1],
        supportTo: convoyMatch[2],
      };
    }
  }

  // Build: "A PAR B" or "F BRE B"
  if (order.includes(" B")) {
    return { type: "build", unit, from: fromLoc };
  }

  // Disband: "A PAR D"
  if (order.includes(" D")) {
    return { type: "disband", unit, from: fromLoc };
  }

  // Retreat: "A PAR R BUR" (retreat to)
  if (order.match(/R [A-Z]{3}/)) {
    const retreatMatch = order.match(/R ([A-Z]{3})/);
    if (retreatMatch) {
      return { type: "retreat", unit, from: fromLoc, to: retreatMatch[1] };
    }
  }

  return null;
}

/**
 * Get coordinates for a province name
 */
function getProvinceCoords(province: string): { x: number; y: number } | null {
  const upper = province.toUpperCase();
  if (PROVINCE_COORDINATES[upper]) {
    return PROVINCE_COORDINATES[upper];
  }
  // Try without coast
  const base = upper.split("/")[0];
  if (PROVINCE_COORDINATES[base]) {
    return PROVINCE_COORDINATES[base];
  }
  return null;
}

export function DiplomacyMap({
  units,
  centers,
  humanPower,
  validMoves,
  selectedOrders = {},
  selectedUnit,
  onUnitClick,
  dragState,
  onDragStart,
  onDragMove,
  onDragEnd,
  onDragCancel,
}: DiplomacyMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [svgContent, setSvgContent] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef<{ x: number; y: number } | null>(null);
  const hasDraggedRef = useRef(false); // Track if user actually dragged (moved significantly)

  // Load the base SVG map
  useEffect(() => {
    fetch("/map.svg")
      .then((res) => res.text())
      .then((text) => {
        let cleaned = text
          .replace(/<\?xml[^?]*\?>/g, "")
          .replace(/<!DOCTYPE[^>]*>/g, "")
          .replace(/xmlns:jdipNS="[^"]*"/g, "");

        cleaned = cleaned.replace(
          /<jdipNS:DISPLAY>[\s\S]*?<\/jdipNS:DISPLAY>/g,
          ""
        );
        cleaned = cleaned.replace(
          /<jdipNS:ORDERDRAWING>[\s\S]*?<\/jdipNS:ORDERDRAWING>/g,
          ""
        );
        cleaned = cleaned.replace(
          /<jdipNS:PROVINCE_DATA>[\s\S]*?<\/jdipNS:PROVINCE_DATA>/g,
          ""
        );

        setSvgContent(cleaned);
      })
      .catch(console.error);
  }, []);

  // Get all units with their power
  const allUnits: Array<{ unit: string; power: string }> = [];
  for (const [power, powerUnits] of Object.entries(units)) {
    for (const unit of powerUnits) {
      allUnits.push({ unit, power });
    }
  }

  // Parse selected orders for visualization
  const parsedOrders: ParsedOrder[] = [];
  for (const [unit, order] of Object.entries(selectedOrders)) {
    const parsed = parseOrder(unit, order);
    if (parsed) {
      parsedOrders.push(parsed);
    }
  }

  const isClickable = (unit: string) => {
    return validMoves && unit in validMoves;
  };

  // Get human power color for order visualization
  const humanColor = UNIT_COLORS[humanPower] || "#00bfff";

  // Minimum drag distance to consider it a drag vs a click (in SVG units)
  const MIN_DRAG_DISTANCE = 15;

  // Drag handlers
  const handlePointerDown = useCallback(
    (unit: string, e: React.PointerEvent) => {
      if (!onDragStart || !svgRef.current) return;
      if (!isClickable(unit)) return;

      e.preventDefault();
      e.stopPropagation();

      const pos = svgPointFromEvent(e.nativeEvent, svgRef.current);
      dragStartPos.current = pos;
      hasDraggedRef.current = false;
      setIsDragging(true);
      onDragStart(unit, pos);

      // Capture pointer for reliable drag tracking
      (e.target as Element).setPointerCapture(e.pointerId);
    },
    [onDragStart, isClickable]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDragging || !onDragMove || !svgRef.current) return;

      const pos = svgPointFromEvent(e.nativeEvent, svgRef.current);

      // Check if we've moved enough to consider it a drag
      if (dragStartPos.current && !hasDraggedRef.current) {
        const dist = Math.hypot(
          pos.x - dragStartPos.current.x,
          pos.y - dragStartPos.current.y
        );
        if (dist > MIN_DRAG_DISTANCE) {
          hasDraggedRef.current = true;
        }
      }

      onDragMove(pos);
    },
    [isDragging, onDragMove]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent) => {
      if (!isDragging || !onDragEnd || !svgRef.current) return;

      const pos = svgPointFromEvent(e.nativeEvent, svgRef.current);

      // If we didn't drag significantly, treat as a click (potential hold)
      if (!hasDraggedRef.current) {
        // Find the unit at the click position to issue hold
        setIsDragging(false);
        dragStartPos.current = null;
        onDragEnd(null); // null signals "dropped on self" which triggers hold
        (e.target as Element).releasePointerCapture?.(e.pointerId);
        return;
      }

      const province = findNearestProvinceFromAll(pos, 60);

      setIsDragging(false);
      dragStartPos.current = null;
      onDragEnd(province);

      // Release pointer capture
      (e.target as Element).releasePointerCapture?.(e.pointerId);
    },
    [isDragging, onDragEnd]
  );

  const handleMapClick = useCallback(
    (e: React.MouseEvent) => {
      // Handle clicks for multi-step operations
      if (!dragState || dragState.phase === "idle" || dragState.phase === "dragging") return;
      if (!svgRef.current || !onDragEnd) return;

      const pos = svgPointFromEvent(e.nativeEvent, svgRef.current);
      const province = findNearestProvinceFromAll(pos, 60);
      onDragEnd(province);
    },
    [dragState, onDragEnd]
  );

  // Check if we're in an active drag state (for showing drag UI)
  const isActiveDrag = dragState && dragState.phase !== "idle";

  return (
    <Card className="h-full overflow-hidden p-0">
      <CardContent className="p-0 h-full">
        <div
          ref={containerRef}
          className="relative w-full overflow-auto"
          style={{ maxHeight: "calc(100vh - 100px)" }}
        >
          <svg
            ref={svgRef}
            viewBox={`0 0 ${MAP_VIEWBOX.width} ${MAP_VIEWBOX.height}`}
            className="w-full h-auto min-w-[600px]"
            style={{ background: "#c5dfea", touchAction: isDragging ? "none" : "auto" }}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={() => {
              setIsDragging(false);
              onDragCancel?.();
            }}
            onClick={handleMapClick}
          >
            {/* Arrow marker definitions */}
            <defs>
              <marker
                id="arrowhead-move"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill={humanColor} />
              </marker>
              <marker
                id="arrowhead-support"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#22c55e" />
              </marker>
              <marker
                id="arrowhead-convoy"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7" />
              </marker>
            </defs>

            {/* Base map from SVG file */}
            {svgContent && (
              <g dangerouslySetInnerHTML={{ __html: extractSvgContent(svgContent) }} />
            )}

            {/* Order visualization layer - draw BEFORE units so arrows go under */}
            <g id="orders">
              {parsedOrders.map((order, idx) => {
                const fromCoords = getProvinceCoords(order.from);
                if (!fromCoords) return null;

                // Move arrow
                if (order.type === "move" && order.to) {
                  const toCoords = getProvinceCoords(order.to);
                  if (!toCoords) return null;

                  return (
                    <g key={`order-${idx}`}>
                      {/* Shadow */}
                      <line
                        x1={fromCoords.x}
                        y1={fromCoords.y}
                        x2={toCoords.x}
                        y2={toCoords.y}
                        stroke="black"
                        strokeWidth="10"
                        opacity="0.3"
                      />
                      {/* Arrow line */}
                      <line
                        x1={fromCoords.x}
                        y1={fromCoords.y}
                        x2={toCoords.x}
                        y2={toCoords.y}
                        stroke={humanColor}
                        strokeWidth="6"
                        markerEnd="url(#arrowhead-move)"
                      />
                      {/* Destination highlight */}
                      <circle
                        cx={toCoords.x}
                        cy={toCoords.y}
                        r="25"
                        fill={humanColor}
                        opacity="0.3"
                      />
                    </g>
                  );
                }

                // Retreat arrow (same as move but different color)
                if (order.type === "retreat" && order.to) {
                  const toCoords = getProvinceCoords(order.to);
                  if (!toCoords) return null;

                  return (
                    <g key={`order-${idx}`}>
                      <line
                        x1={fromCoords.x}
                        y1={fromCoords.y}
                        x2={toCoords.x}
                        y2={toCoords.y}
                        stroke="black"
                        strokeWidth="10"
                        opacity="0.3"
                      />
                      <line
                        x1={fromCoords.x}
                        y1={fromCoords.y}
                        x2={toCoords.x}
                        y2={toCoords.y}
                        stroke="#ef4444"
                        strokeWidth="6"
                        strokeDasharray="15,5"
                        markerEnd="url(#arrowhead-move)"
                      />
                    </g>
                  );
                }

                // Hold - octagon around unit
                if (order.type === "hold") {
                  return (
                    <g key={`order-${idx}`}>
                      <polygon
                        points={octagonPoints(fromCoords.x, fromCoords.y, 35)}
                        fill="none"
                        stroke={humanColor}
                        strokeWidth="5"
                        opacity="0.8"
                      />
                    </g>
                  );
                }

                // Support - dashed arrow to supported unit or move
                if (order.type === "support") {
                  if (order.supportTo && order.supportFrom) {
                    // Support move: draw arrow from supporter to the destination
                    const supportToCoords = getProvinceCoords(order.supportTo);
                    if (!supportToCoords) return null;

                    return (
                      <g key={`order-${idx}`}>
                        <line
                          x1={fromCoords.x}
                          y1={fromCoords.y}
                          x2={supportToCoords.x}
                          y2={supportToCoords.y}
                          stroke="black"
                          strokeWidth="8"
                          opacity="0.2"
                        />
                        <line
                          x1={fromCoords.x}
                          y1={fromCoords.y}
                          x2={supportToCoords.x}
                          y2={supportToCoords.y}
                          stroke="#22c55e"
                          strokeWidth="5"
                          strokeDasharray="10,5"
                          markerEnd="url(#arrowhead-support)"
                        />
                      </g>
                    );
                  } else if (order.supportTarget) {
                    // Support hold: draw line to supported unit
                    const targetCoords = getProvinceCoords(order.supportTarget);
                    if (!targetCoords) return null;

                    return (
                      <g key={`order-${idx}`}>
                        <line
                          x1={fromCoords.x}
                          y1={fromCoords.y}
                          x2={targetCoords.x}
                          y2={targetCoords.y}
                          stroke="black"
                          strokeWidth="8"
                          opacity="0.2"
                        />
                        <line
                          x1={fromCoords.x}
                          y1={fromCoords.y}
                          x2={targetCoords.x}
                          y2={targetCoords.y}
                          stroke="#22c55e"
                          strokeWidth="5"
                          strokeDasharray="10,5"
                        />
                        {/* Support hold indicator at target */}
                        <polygon
                          points={octagonPoints(targetCoords.x, targetCoords.y, 30)}
                          fill="none"
                          stroke="#22c55e"
                          strokeWidth="3"
                          strokeDasharray="8,4"
                          opacity="0.7"
                        />
                      </g>
                    );
                  }
                }

                // Convoy - purple dashed line
                if (order.type === "convoy" && order.supportFrom && order.supportTo) {
                  const convoyFromCoords = getProvinceCoords(order.supportFrom);
                  const convoyToCoords = getProvinceCoords(order.supportTo);
                  if (!convoyFromCoords || !convoyToCoords) return null;

                  return (
                    <g key={`order-${idx}`}>
                      {/* Triangle at convoy location */}
                      <polygon
                        points={trianglePoints(fromCoords.x, fromCoords.y, 30)}
                        fill="none"
                        stroke="#a855f7"
                        strokeWidth="4"
                        strokeDasharray="10,5"
                      />
                      {/* Line showing convoy route */}
                      <line
                        x1={convoyFromCoords.x}
                        y1={convoyFromCoords.y}
                        x2={convoyToCoords.x}
                        y2={convoyToCoords.y}
                        stroke="#a855f7"
                        strokeWidth="4"
                        strokeDasharray="15,5"
                        opacity="0.5"
                      />
                    </g>
                  );
                }

                // Build - concentric circles
                if (order.type === "build") {
                  return (
                    <g key={`order-${idx}`}>
                      <circle cx={fromCoords.x} cy={fromCoords.y} r="20" fill="none" stroke="#22c55e" strokeWidth="3" />
                      <circle cx={fromCoords.x} cy={fromCoords.y} r="30" fill="none" stroke="#22c55e" strokeWidth="3" />
                      <circle cx={fromCoords.x} cy={fromCoords.y} r="40" fill="none" stroke="#22c55e" strokeWidth="3" />
                    </g>
                  );
                }

                // Disband - X mark
                if (order.type === "disband") {
                  return (
                    <g key={`order-${idx}`}>
                      <line
                        x1={fromCoords.x - 25}
                        y1={fromCoords.y - 25}
                        x2={fromCoords.x + 25}
                        y2={fromCoords.y + 25}
                        stroke="#ef4444"
                        strokeWidth="8"
                      />
                      <line
                        x1={fromCoords.x + 25}
                        y1={fromCoords.y - 25}
                        x2={fromCoords.x - 25}
                        y2={fromCoords.y + 25}
                        stroke="#ef4444"
                        strokeWidth="8"
                      />
                    </g>
                  );
                }

                return null;
              })}
            </g>

            {/* Drag order layer - between orders and units */}
            {dragState && (
              <DragOrderLayer
                dragState={dragState}
                humanPower={humanPower}
                allUnits={allUnits}
              />
            )}

            {/* Unit overlays */}
            <g id="units">
              {allUnits.map(({ unit, power }) => {
                const coords = getUnitCoordinates(unit);
                if (!coords) return null;

                const fleet = isFleet(unit);
                const color = UNIT_COLORS[power] || "#888";
                const isHuman = power === humanPower;
                const clickable = isClickable(unit);
                const isSelected = selectedUnit === unit || (dragState?.phase !== "idle" && dragState?.unit === unit);
                const hasOrder = unit in selectedOrders;

                return (
                  <g
                    key={unit}
                    transform={`translate(${coords.x - 20}, ${coords.y - 10})`}
                    onClick={() => !isDragging && clickable && onUnitClick?.(unit)}
                    onPointerDown={(e) => handlePointerDown(unit, e)}
                    style={{ cursor: clickable ? "grab" : "default" }}
                  >
                    {/* Selection highlight */}
                    {isSelected && (
                      <rect
                        x="-4"
                        y="-4"
                        width="48"
                        height="28"
                        rx="8"
                        fill="none"
                        stroke="#fbbf24"
                        strokeWidth="3"
                        className="animate-pulse"
                      />
                    )}
                    {/* Human unit with order highlight */}
                    {isHuman && hasOrder && !isSelected && (
                      <rect
                        x="-2"
                        y="-2"
                        width="44"
                        height="24"
                        rx="6"
                        fill="none"
                        stroke="#22c55e"
                        strokeWidth="3"
                      />
                    )}
                    {/* Human unit without order */}
                    {isHuman && !hasOrder && !isSelected && (
                      <rect
                        x="-2"
                        y="-2"
                        width="44"
                        height="24"
                        rx="6"
                        fill="none"
                        stroke="#f97316"
                        strokeWidth="2"
                        strokeDasharray="4,2"
                        opacity="0.8"
                      />
                    )}
                    {/* Unit shadow */}
                    <rect
                      x="3"
                      y="3"
                      width="40"
                      height="20"
                      rx="4"
                      fill="black"
                      opacity="0.4"
                    />
                    {/* Unit body */}
                    <rect
                      x="0"
                      y="0"
                      width="40"
                      height="20"
                      rx="4"
                      fill={color}
                      stroke="black"
                      strokeWidth="1"
                    />
                    {/* Unit type indicator */}
                    <text
                      x="20"
                      y="15"
                      textAnchor="middle"
                      fontSize="12"
                      fontWeight="bold"
                      fill="black"
                    >
                      {fleet ? "F" : "A"}
                    </text>
                  </g>
                );
              })}
            </g>

            {/* Supply center markers */}
            <g id="supply-centers">
              {Object.entries(centers).map(([power, powerCenters]) =>
                powerCenters.map((center) => {
                  const coords = PROVINCE_COORDINATES[center.toUpperCase()];
                  if (!coords) return null;

                  const hasUnit = allUnits.some((u) => {
                    const loc = parseUnitLocation(u.unit);
                    return loc === center.toUpperCase();
                  });
                  if (hasUnit) return null;

                  const color = UNIT_COLORS[power] || "#888";

                  return (
                    <g
                      key={`sc-${center}`}
                      transform={`translate(${coords.x}, ${coords.y})`}
                    >
                      <circle
                        r="8"
                        fill={color}
                        stroke="black"
                        strokeWidth="1.5"
                      />
                      <circle r="4" fill="black" opacity="0.3" />
                    </g>
                  );
                })
              )}
            </g>
          </svg>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Generate octagon points for hold indicator
 */
function octagonPoints(cx: number, cy: number, r: number): string {
  const points: string[] = [];
  for (let i = 0; i < 8; i++) {
    const angle = (Math.PI / 4) * i - Math.PI / 8;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    points.push(`${x},${y}`);
  }
  return points.join(" ");
}

/**
 * Generate triangle points for convoy indicator
 */
function trianglePoints(cx: number, cy: number, r: number): string {
  const points: string[] = [];
  for (let i = 0; i < 3; i++) {
    const angle = (Math.PI * 2 / 3) * i - Math.PI / 2;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    points.push(`${x},${y}`);
  }
  return points.join(" ");
}

/**
 * Extract just the visual content from the SVG
 */
function extractSvgContent(svg: string): string {
  const defsEnd = svg.indexOf("</defs>");
  if (defsEnd === -1) return "";

  const endSvg = svg.lastIndexOf("</svg>");
  if (endSvg === -1) return "";

  let content = svg.substring(defsEnd + 7, endSvg);

  const defsStart = svg.indexOf("<defs>");
  if (defsStart !== -1) {
    const defs = svg.substring(defsStart, defsEnd + 7);
    content = defs + content;
  }

  return content;
}
