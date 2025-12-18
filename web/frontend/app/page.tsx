"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { gameApi } from "@/lib/api";
import { useGameCreationStore } from "@/lib/store";
import { POWERS, POWER_COLORS, type Power } from "@/lib/types";

export default function HomePage() {
  const router = useRouter();
  const {
    selectedPower,
    selectedAdapter,
    availableAdapters,
    isCreating,
    error,
    setSelectedPower,
    setSelectedAdapter,
    setAvailableAdapters,
    setCreating,
    setError,
  } = useGameCreationStore();

  // Fetch available adapters
  const { data: adapters } = useQuery({
    queryKey: ["adapters"],
    queryFn: gameApi.getAdapters,
  });

  useEffect(() => {
    if (adapters) {
      setAvailableAdapters(adapters);
    }
  }, [adapters, setAvailableAdapters]);

  // Create game mutation
  const createGameMutation = useMutation({
    mutationFn: gameApi.createGame,
    onMutate: () => {
      setCreating(true);
      setError(null);
    },
    onSuccess: (game) => {
      router.push(`/game/${game.id}`);
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to create game");
      setCreating(false);
    },
  });

  const handleCreateGame = () => {
    createGameMutation.mutate({
      human_power: selectedPower,
      adapter_name: selectedAdapter,
    });
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="max-w-2xl w-full space-y-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Diplomacy AI</h1>
          <p className="text-muted-foreground">
            Play the classic game of strategy against AI opponents
          </p>
        </div>

        {/* Game Creation Card */}
        <Card>
          <CardHeader>
            <CardTitle>New Game</CardTitle>
            <CardDescription>
              Choose your power and AI difficulty to start a new game
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Power Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Select Your Power</label>
              <Select value={selectedPower} onValueChange={setSelectedPower}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a power..." />
                </SelectTrigger>
                <SelectContent>
                  {POWERS.map((power) => (
                    <SelectItem key={power} value={power}>
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{
                            backgroundColor: POWER_COLORS[power as Power],
                          }}
                        />
                        {power}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Adapter Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">AI Difficulty</label>
              <Select
                value={selectedAdapter ?? "base"}
                onValueChange={(v) =>
                  setSelectedAdapter(v === "base" ? null : v)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Choose difficulty..." />
                </SelectTrigger>
                <SelectContent>
                  {availableAdapters.map((adapter) => (
                    <SelectItem
                      key={adapter.id ?? "base"}
                      value={adapter.id ?? "base"}
                    >
                      <div className="flex flex-col">
                        <span>{adapter.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {adapter.description}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md text-destructive text-sm">
                {error}
              </div>
            )}

            {/* Create Button */}
            <Button
              onClick={handleCreateGame}
              disabled={isCreating}
              className="w-full"
              size="lg"
            >
              {isCreating ? "Creating Game..." : "Start Game"}
            </Button>
          </CardContent>
        </Card>

        {/* Game Info */}
        <Card>
          <CardContent className="pt-6">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <h3 className="font-medium mb-2">How to Play</h3>
                <ul className="text-muted-foreground space-y-1 list-disc list-inside">
                  <li>Choose orders for your units each turn</li>
                  <li>AI controls all other powers</li>
                  <li>Capture supply centers to win</li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium mb-2">Phase Types</h3>
                <div className="space-y-1">
                  <Badge variant="outline" className="mr-2">
                    Movement
                  </Badge>
                  <span className="text-muted-foreground">
                    Move & support units
                  </span>
                </div>
                <div className="space-y-1 mt-1">
                  <Badge variant="outline" className="mr-2">
                    Retreat
                  </Badge>
                  <span className="text-muted-foreground">
                    Retreat dislodged units
                  </span>
                </div>
                <div className="space-y-1 mt-1">
                  <Badge variant="outline" className="mr-2">
                    Build
                  </Badge>
                  <span className="text-muted-foreground">
                    Build or disband units
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
