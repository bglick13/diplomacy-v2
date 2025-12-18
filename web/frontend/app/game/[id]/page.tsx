"use client";

import { useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { gameApi } from "@/lib/api";
import { useGameStore, useOrdersArray } from "@/lib/store";
import { PhaseDisplay } from "@/components/PhaseDisplay";
import { PowerStatus } from "@/components/PowerStatus";
import { OrderInput } from "@/components/OrderInput";
import { GameBoard } from "@/components/GameBoard";
import { POWER_COLORS, type Power } from "@/lib/types";

export default function GamePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const gameId = params.id;

  const { setGame, clearOrders, setSubmitting, setError, error, selectedOrders } = useGameStore();
  const ordersArray = useOrdersArray();

  // Fetch game state
  const {
    data: game,
    isLoading,
    error: fetchError,
  } = useQuery({
    queryKey: ["game", gameId],
    queryFn: () => gameApi.getGame(gameId),
    refetchInterval: false,
  });

  // Update store when game data changes
  useEffect(() => {
    if (game) {
      setGame(game);
    }
  }, [game, setGame]);

  // Submit orders mutation
  const submitMutation = useMutation({
    mutationFn: (orders: string[]) => gameApi.submitOrders(gameId, orders),
    onMutate: () => {
      setSubmitting(true);
      setError(null);
    },
    onSuccess: (newState) => {
      setGame(newState);
      clearOrders();
      queryClient.setQueryData(["game", gameId], newState);
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : "Failed to submit orders");
    },
    onSettled: () => {
      setSubmitting(false);
    },
  });

  const handleSubmit = () => {
    submitMutation.mutate(ordersArray);
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-muted-foreground">Loading game...</div>
      </div>
    );
  }

  // Error state
  if (fetchError || !game) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="pt-6 text-center space-y-4">
            <p className="text-destructive">
              {fetchError instanceof Error
                ? fetchError.message
                : "Game not found"}
            </p>
            <Button asChild>
              <Link href="/">Back to Home</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b p-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link
              href="/"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              &larr; Home
            </Link>
            <PhaseDisplay phase={game.phase} year={game.year} />
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">Playing as</span>
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{
                  backgroundColor: POWER_COLORS[game.human_power as Power],
                }}
              />
              <span className="font-medium">{game.human_power}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Error Display */}
      {error && (
        <div className="p-3 bg-destructive/10 border-b border-destructive/20 text-destructive text-sm text-center">
          {error}
        </div>
      )}

      {/* Game Complete State */}
      {game.is_done && (
        <div className="p-4 border-b">
          <div className="text-center space-y-3">
            <h2 className="text-xl font-bold">Game Complete!</h2>
            <div className="flex items-center justify-center gap-4">
              <Badge variant="outline" className="px-3 py-1">
                Final Year: {game.year}
              </Badge>
              {game.trajectories_collected && (
                <Badge className="px-3 py-1 bg-green-600 text-white">
                  {game.trajectories_collected} training samples collected
                </Badge>
              )}
            </div>
            <Button asChild size="sm">
              <Link href="/">Start New Game</Link>
            </Button>
          </div>
        </div>
      )}

      {/* Main Content - Map + Orders Side by Side */}
      <main className="flex-1 flex min-h-0 overflow-hidden">
        {/* Map Area */}
        <div className="flex-1 p-2 min-w-0">
          <GameBoard
            units={game.all_units}
            centers={game.all_centers}
            humanPower={game.human_power}
            validMoves={game.valid_moves}
            selectedOrders={selectedOrders}
          />
        </div>

        {/* Orders Sidebar */}
        {!game.is_done && (
          <div className="w-72 flex-shrink-0 border-l flex flex-col overflow-hidden">
            {/* Compact Power Rankings */}
            <div className="p-3 border-b flex-shrink-0">
              <div className="text-xs font-medium text-muted-foreground mb-2">Rankings</div>
              <div className="space-y-1">
                {game.board_context.power_rankings.map(([power, centers], index) => {
                  const isHuman = power === game.human_power;
                  const unitCount = game.all_units[power]?.length ?? 0;
                  return (
                    <div
                      key={power}
                      className={`flex items-center gap-1.5 text-xs ${isHuman ? "font-semibold" : ""}`}
                    >
                      <span className="text-muted-foreground w-3">{index + 1}.</span>
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: POWER_COLORS[power as Power] }}
                      />
                      <span className="flex-1 truncate">{power}</span>
                      <span className="text-muted-foreground">
                        {centers}/{unitCount}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Order Controls */}
            <div className="flex-1 overflow-y-auto p-3">
              <OrderInput
                validMoves={game.valid_moves}
                onSubmit={handleSubmit}
              />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t p-2 flex-shrink-0">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Game ID: {game.id}</span>
          <span>
            {game.board_context.my_units.length} units &bull;{" "}
            {game.board_context.my_centers.length} supply centers
          </span>
        </div>
      </footer>
    </div>
  );
}
