/**
 * Map coordinate data for the Diplomacy game board.
 * Coordinates are from the standard.svg file.
 */

// Province coordinates for unit placement
export const PROVINCE_COORDINATES: Record<string, { x: number; y: number }> = {
  ADR: { x: 793.5, y: 1048.0 },
  AEG: { x: 1043.5, y: 1230.0 },
  ALB: { x: 906.5, y: 1113.0 },
  ANK: { x: 1301.5, y: 1110.0 },
  APU: { x: 791.5, y: 1106.0 },
  ARM: { x: 1484.5, y: 1090.0 },
  BAL: { x: 878.5, y: 610.0 },
  BAR: { x: 1162.5, y: 73.0 },
  BEL: { x: 561.5, y: 753.0 },
  BER: { x: 771.5, y: 690.0 },
  BLA: { x: 1233.5, y: 1000.0 },
  BOH: { x: 806.5, y: 814.0 },
  BOT: { x: 941.5, y: 485.0 },
  BRE: { x: 404.5, y: 819.0 },
  BUD: { x: 950.5, y: 904.0 },
  BUL: { x: 1048.5, y: 1068.0 },
  "BUL/EC": { x: 1127.0, y: 1067.0 },
  "BUL/SC": { x: 1070.0, y: 1140.0 },
  BUR: { x: 559.5, y: 871.0 },
  CLY: { x: 436.5, y: 492.0 },
  CON: { x: 1145.5, y: 1137.0 },
  DEN: { x: 703.5, y: 587.0 },
  EAS: { x: 1218.5, y: 1311.0 },
  EDI: { x: 473.5, y: 514.0 },
  ENG: { x: 394.5, y: 751.0 },
  FIN: { x: 988.5, y: 380.0 },
  GAL: { x: 999.5, y: 831.0 },
  GAS: { x: 422.5, y: 912.0 },
  GOL: { x: 556.0, y: 1060.0 },
  GRE: { x: 966.5, y: 1190.0 },
  HEL: { x: 651.5, y: 631.0 },
  HOL: { x: 596.5, y: 711.0 },
  ION: { x: 846.5, y: 1286.0 },
  IRI: { x: 335.5, y: 661.0 },
  KIE: { x: 683.5, y: 701.0 },
  LON: { x: 488.5, y: 675.0 },
  LVN: { x: 1025.5, y: 567.0 },
  LVP: { x: 450.5, y: 576.0 },
  LYO: { x: 514.3, y: 1055.0 },
  MAO: { x: 141.8, y: 835.3 },
  MAR: { x: 524.5, y: 975.0 },
  MOS: { x: 1200.5, y: 590.0 },
  MUN: { x: 693.5, y: 828.0 },
  NAF: { x: 325.5, y: 1281.0 },
  NAO: { x: 180.1, y: 288.2 },
  NAP: { x: 806.5, y: 1170.0 },
  NTH: { x: 553.5, y: 560.0 },
  NWG: { x: 652.7, y: 181.8 },
  NWY: { x: 703.5, y: 410.0 },
  PAR: { x: 488.5, y: 845.0 },
  PIC: { x: 523.5, y: 781.0 },
  PIE: { x: 630.5, y: 968.0 },
  POR: { x: 181.5, y: 1013.0 },
  PRU: { x: 865.5, y: 690.0 },
  ROM: { x: 731.5, y: 1102.0 },
  RUH: { x: 636.5, y: 779.0 },
  RUM: { x: 1096.5, y: 967.0 },
  SER: { x: 933.5, y: 1050.0 },
  SEV: { x: 1284.5, y: 845.0 },
  SIL: { x: 832.5, y: 769.0 },
  SKA: { x: 735.5, y: 518.0 },
  SMY: { x: 1253.5, y: 1210.0 },
  SPA: { x: 335.5, y: 1039.0 },
  "SPA/NC": { x: 289.0, y: 965.0 },
  "SPA/SC": { x: 291.0, y: 1166.0 },
  STP: { x: 1166.5, y: 405.0 },
  "STP/NC": { x: 1218.0, y: 222.0 },
  "STP/SC": { x: 1066.0, y: 487.0 },
  SWE: { x: 829.5, y: 459.0 },
  SYR: { x: 1452.5, y: 1206.0 },
  TRI: { x: 825.5, y: 996.0 },
  TUN: { x: 622.5, y: 1300.0 },
  TUS: { x: 686.5, y: 1034.0 },
  TYR: { x: 742.5, y: 904.0 },
  UKR: { x: 1124.5, y: 800.0 },
  VEN: { x: 707.5, y: 994.0 },
  VIE: { x: 855.5, y: 864.0 },
  WAL: { x: 428.5, y: 658.0 },
  WAR: { x: 983.5, y: 740.0 },
  YOR: { x: 492.5, y: 616.0 },
};

// Map viewBox dimensions
export const MAP_VIEWBOX = {
  width: 1835,
  height: 1360,
};

// Power colors matching the SVG definitions
export const MAP_POWER_COLORS: Record<string, string> = {
  AUSTRIA: "#c48f85",
  ENGLAND: "#8a2be2", // darkviolet
  FRANCE: "#4169e1", // royalblue
  GERMANY: "#a08a75",
  ITALY: "#228b22", // forestgreen
  RUSSIA: "#757d91",
  TURKEY: "#b9a61c",
};

// Unit colors (slightly different for visibility)
export const UNIT_COLORS: Record<string, string> = {
  AUSTRIA: "#ef4444", // red
  ENGLAND: "#9370db", // mediumpurple
  FRANCE: "#00bfff", // deepskyblue
  GERMANY: "#696969", // dimgray
  ITALY: "#808000", // olive
  RUSSIA: "#ffffff", // white
  TURKEY: "#ffd700", // gold
};

/**
 * Parse a unit string like "A PAR" or "F BRE" to get location
 */
export function parseUnitLocation(unit: string): string {
  // Handle formats like "A PAR", "F BRE", "F STP/SC"
  const parts = unit.split(" ");
  if (parts.length >= 2) {
    return parts.slice(1).join(" ").toUpperCase();
  }
  return unit.toUpperCase();
}

/**
 * Check if unit is a fleet
 */
export function isFleet(unit: string): boolean {
  return unit.startsWith("F ");
}

/**
 * Get coordinates for a unit, handling coastal variants
 */
export function getUnitCoordinates(
  unit: string
): { x: number; y: number } | null {
  const location = parseUnitLocation(unit);

  // Direct match
  if (PROVINCE_COORDINATES[location]) {
    return PROVINCE_COORDINATES[location];
  }

  // Try without coast suffix for display
  const baseLocation = location.split("/")[0];
  if (PROVINCE_COORDINATES[baseLocation]) {
    return PROVINCE_COORDINATES[baseLocation];
  }

  return null;
}
