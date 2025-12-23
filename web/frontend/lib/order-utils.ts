/**
 * Order parsing and destination extraction utilities for drag-and-drop input.
 */

export interface ParsedOrder {
  raw: string;
  type: "move" | "hold" | "support_hold" | "support_move" | "convoy" | "build" | "disband" | "retreat";
  unit: string; // e.g., "A PAR"
  unitType: "A" | "F";
  from: string; // Province where unit is
  to?: string; // Destination for move/retreat
  targetUnit?: string; // Unit being supported (e.g., "A BUR")
  targetFrom?: string; // Where supported unit is
  targetTo?: string; // Where supported unit is moving
  convoyArmy?: string; // Army being convoyed
  convoyTo?: string; // Convoy destination
}

/**
 * Parse an order string into structured data.
 */
export function parseOrderString(orderStr: string): ParsedOrder | null {
  const trimmed = orderStr.trim();

  // Extract unit and type
  const unitMatch = trimmed.match(/^([AF])\s+([A-Z]{3}(?:\/[A-Z]{2})?)/);
  if (!unitMatch) {
    // Handle WAIVE
    if (trimmed === "WAIVE") {
      return {
        raw: trimmed,
        type: "build",
        unit: "WAIVE",
        unitType: "A",
        from: "",
      };
    }
    return null;
  }

  const unitType = unitMatch[1] as "A" | "F";
  const from = unitMatch[2];
  const unit = `${unitType} ${from}`;

  // Hold: "A PAR H"
  if (trimmed.endsWith(" H")) {
    return { raw: trimmed, type: "hold", unit, unitType, from };
  }

  // Build: "A PAR B"
  if (trimmed.endsWith(" B")) {
    return { raw: trimmed, type: "build", unit, unitType, from };
  }

  // Disband: "A PAR D"
  if (trimmed.endsWith(" D")) {
    return { raw: trimmed, type: "disband", unit, unitType, from };
  }

  // Retreat: "A PAR R BUR"
  const retreatMatch = trimmed.match(/R\s+([A-Z]{3}(?:\/[A-Z]{2})?)$/);
  if (retreatMatch) {
    return { raw: trimmed, type: "retreat", unit, unitType, from, to: retreatMatch[1] };
  }

  // Convoy: "F NTH C A LON - BEL"
  const convoyMatch = trimmed.match(/C\s+([AF])\s+([A-Z]{3})\s+-\s+([A-Z]{3}(?:\/[A-Z]{2})?)$/);
  if (convoyMatch) {
    return {
      raw: trimmed,
      type: "convoy",
      unit,
      unitType,
      from,
      convoyArmy: `${convoyMatch[1]} ${convoyMatch[2]}`,
      targetFrom: convoyMatch[2],
      convoyTo: convoyMatch[3],
    };
  }

  // Support move: "A PAR S A BUR - MAR"
  const supportMoveMatch = trimmed.match(/S\s+([AF])\s+([A-Z]{3}(?:\/[A-Z]{2})?)\s+-\s+([A-Z]{3}(?:\/[A-Z]{2})?)$/);
  if (supportMoveMatch) {
    return {
      raw: trimmed,
      type: "support_move",
      unit,
      unitType,
      from,
      targetUnit: `${supportMoveMatch[1]} ${supportMoveMatch[2]}`,
      targetFrom: supportMoveMatch[2],
      targetTo: supportMoveMatch[3],
    };
  }

  // Support hold: "A PAR S A BUR"
  const supportHoldMatch = trimmed.match(/S\s+([AF])\s+([A-Z]{3}(?:\/[A-Z]{2})?)$/);
  if (supportHoldMatch) {
    return {
      raw: trimmed,
      type: "support_hold",
      unit,
      unitType,
      from,
      targetUnit: `${supportHoldMatch[1]} ${supportHoldMatch[2]}`,
      targetFrom: supportHoldMatch[2],
    };
  }

  // Move: "A PAR - BUR"
  const moveMatch = trimmed.match(/-\s+([A-Z]{3}(?:\/[A-Z]{2})?)$/);
  if (moveMatch) {
    return { raw: trimmed, type: "move", unit, unitType, from, to: moveMatch[1] };
  }

  return null;
}

/**
 * Parse all valid moves for a unit into structured data.
 */
export function parseValidMoves(validMoves: string[]): ParsedOrder[] {
  return validMoves.map(parseOrderString).filter((p): p is ParsedOrder => p !== null);
}

/**
 * Get all valid move destinations (provinces) for a unit.
 */
export function getMoveDestinations(validMoves: string[]): string[] {
  const destinations = new Set<string>();
  for (const order of validMoves) {
    const parsed = parseOrderString(order);
    if (parsed?.type === "move" && parsed.to) {
      destinations.add(parsed.to);
    }
  }
  return Array.from(destinations);
}

/**
 * Check if a unit can hold.
 */
export function canHold(validMoves: string[]): boolean {
  return validMoves.some((order) => order.endsWith(" H"));
}

/**
 * Get the hold order string if available.
 */
export function getHoldOrder(validMoves: string[]): string | null {
  return validMoves.find((order) => order.endsWith(" H")) ?? null;
}

/**
 * Get all units (locations) that can be supported from this unit.
 * Returns location strings (e.g., "PAR", "BUR").
 */
export function getSupportableLocations(validMoves: string[]): string[] {
  const locations = new Set<string>();
  for (const order of validMoves) {
    const parsed = parseOrderString(order);
    if ((parsed?.type === "support_hold" || parsed?.type === "support_move") && parsed.targetFrom) {
      locations.add(parsed.targetFrom);
    }
  }
  return Array.from(locations);
}

/**
 * Get the unit type at a given location based on valid moves.
 */
export function getUnitTypeAtLocation(validMoves: string[], location: string): "A" | "F" | null {
  for (const order of validMoves) {
    const parsed = parseOrderString(order);
    if ((parsed?.type === "support_hold" || parsed?.type === "support_move") && parsed.targetFrom === location) {
      const match = order.match(/S\s+([AF])\s+/);
      if (match) return match[1] as "A" | "F";
    }
    if (parsed?.type === "convoy" && parsed.targetFrom === location) {
      return "A"; // Convoys only move armies
    }
  }
  return null;
}

/**
 * Get all valid destinations when supporting a unit at the given location.
 * This includes the location itself for support-hold.
 */
export function getSupportMoveDestinations(validMoves: string[], targetLocation: string): string[] {
  const destinations = new Set<string>();
  for (const order of validMoves) {
    const parsed = parseOrderString(order);
    if (parsed?.type === "support_hold" && parsed.targetFrom === targetLocation) {
      // Support hold - the "destination" is the same location (unit stays)
      destinations.add(targetLocation);
    }
    if (parsed?.type === "support_move" && parsed.targetFrom === targetLocation && parsed.targetTo) {
      destinations.add(parsed.targetTo);
    }
  }
  return Array.from(destinations);
}

/**
 * Get armies that can be convoyed by this fleet.
 * Returns location strings where armies are.
 */
export function getConvoyableArmyLocations(validMoves: string[]): string[] {
  const locations = new Set<string>();
  for (const order of validMoves) {
    const parsed = parseOrderString(order);
    if (parsed?.type === "convoy" && parsed.targetFrom) {
      locations.add(parsed.targetFrom);
    }
  }
  return Array.from(locations);
}

/**
 * Get convoy destinations for a specific army.
 */
export function getConvoyDestinations(validMoves: string[], armyLocation: string): string[] {
  const destinations = new Set<string>();
  for (const order of validMoves) {
    const parsed = parseOrderString(order);
    if (parsed?.type === "convoy" && parsed.targetFrom === armyLocation && parsed.convoyTo) {
      destinations.add(parsed.convoyTo);
    }
  }
  return Array.from(destinations);
}

/**
 * Build an order string from drag-and-drop action.
 */
export function buildOrderString(
  unit: string,
  action:
    | { type: "move"; to: string }
    | { type: "hold" }
    | { type: "support_hold"; targetUnit: string }
    | { type: "support_move"; targetUnit: string; targetTo: string }
    | { type: "convoy"; armyUnit: string; to: string }
): string {
  switch (action.type) {
    case "move":
      return `${unit} - ${action.to}`;
    case "hold":
      return `${unit} H`;
    case "support_hold":
      return `${unit} S ${action.targetUnit}`;
    case "support_move":
      return `${unit} S ${action.targetUnit} - ${action.targetTo}`;
    case "convoy":
      return `${unit} C ${action.armyUnit} - ${action.to}`;
  }
}

/**
 * Find the exact order string from validMoves that matches the action.
 * This handles edge cases where the built string might differ slightly.
 */
export function findMatchingOrder(
  validMoves: string[],
  unit: string,
  action:
    | { type: "move"; to: string }
    | { type: "hold" }
    | { type: "support_hold"; targetLocation: string }
    | { type: "support_move"; targetLocation: string; targetTo: string }
    | { type: "convoy"; armyLocation: string; to: string }
): string | null {
  for (const order of validMoves) {
    const parsed = parseOrderString(order);
    if (!parsed || parsed.unit !== unit) continue;

    switch (action.type) {
      case "move":
        if (parsed.type === "move" && parsed.to === action.to) return order;
        break;
      case "hold":
        if (parsed.type === "hold") return order;
        break;
      case "support_hold":
        if (parsed.type === "support_hold" && parsed.targetFrom === action.targetLocation) return order;
        break;
      case "support_move":
        if (parsed.type === "support_move" && parsed.targetFrom === action.targetLocation && parsed.targetTo === action.targetTo) return order;
        break;
      case "convoy":
        if (parsed.type === "convoy" && parsed.targetFrom === action.armyLocation && parsed.convoyTo === action.to) return order;
        break;
    }
  }
  return null;
}

/**
 * Get base province from a potentially coastal province.
 */
export function getBaseProvince(province: string): string {
  return province.split("/")[0];
}

/**
 * Check if a province has multiple coasts.
 */
export function hasMultipleCoasts(province: string): boolean {
  const base = getBaseProvince(province);
  return ["SPA", "STP", "BUL"].includes(base);
}

/**
 * Get coastal variants for a province.
 */
export function getCoastalVariants(province: string): string[] {
  const base = getBaseProvince(province);
  switch (base) {
    case "SPA":
      return ["SPA/NC", "SPA/SC"];
    case "STP":
      return ["STP/NC", "STP/SC"];
    case "BUL":
      return ["BUL/EC", "BUL/SC"];
    default:
      return [base];
  }
}
