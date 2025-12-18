"use client";

import { DiplomacyMap } from "./DiplomacyMap";

interface GameBoardProps {
  units: Record<string, string[]>;
  centers: Record<string, string[]>;
  humanPower: string;
  compactMapView?: string;
  validMoves?: Record<string, string[]>;
  selectedOrders?: Record<string, string>;
  selectedUnit?: string | null;
  onUnitClick?: (unit: string) => void;
}

/**
 * Game board visualization using the SVG map.
 */
export function GameBoard({
  units,
  centers,
  humanPower,
  validMoves,
  selectedOrders,
  selectedUnit,
  onUnitClick,
}: GameBoardProps) {
  return (
    <DiplomacyMap
      units={units}
      centers={centers}
      humanPower={humanPower}
      validMoves={validMoves}
      selectedOrders={selectedOrders}
      selectedUnit={selectedUnit}
      onUnitClick={onUnitClick}
    />
  );
}
