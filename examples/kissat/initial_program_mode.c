/*
 * Kissat 4.0.4 (commit 8af8e56f174b778aef3aa45af9f739b2a5f492c2) src/mode.c,
 * with an EVOLVE-BLOCK around the mode-switching policy for openevolve-bedrock
 * (examples/kissat, KISSAT_EVAL_SOURCE=mode.c). Kissat is Copyright (c)
 * 2019-2025 Armin Biere, Mathias Fleury, Florian Pollitt and contributors, MIT
 * License: see LICENSE.kissat.
 *
 * The evaluator builds Kissat with this file as src/mode.c. Only the two
 * functions inside the EVOLVE-BLOCK may change; their signatures, and the rest
 * of this file, must stay as they are. Differences from stock: the two
 * functions are moved together, and the block drops update_mode_limit's
 * verbose-only logging.
 */

#include "bump.h"
#include "decide.h"
#include "inline.h"
#include "print.h"
#include "report.h"
#include "resources.h"
#include "restart.h"

#include <inttypes.h>

#ifndef QUIET

static const char *mode_string (kissat *solver) {
  return solver->stable ? "stable" : "focused";
}

#endif

void kissat_init_mode_limit (kissat *solver) {
  limits *limits = &solver->limits;

  if (GET_OPTION (stable) == 1) {
    assert (!solver->stable);

    const uint64_t conflicts_delta = GET_OPTION (modeinit);
    const uint64_t conflicts_limit = CONFLICTS + conflicts_delta;

    assert (conflicts_limit);

    limits->mode.conflicts = conflicts_limit;
    limits->mode.ticks = 0;
    limits->mode.count = 0;

    kissat_very_verbose (solver,
                         "initial %s mode switching limit "
                         "at %s after %s conflicts",
                         mode_string (solver),
                         FORMAT_COUNT (conflicts_limit),
                         FORMAT_COUNT (conflicts_delta));

    solver->mode.ticks = solver->statistics.search_ticks;
#ifndef QUIET
    solver->mode.conflicts = CONFLICTS;
#ifdef METRICS
    solver->mode.propagations = solver->statistics.search_propagations;
#endif
    // clang-format off
      solver->mode.entered = kissat_process_time ();
      kissat_very_verbose (solver,
        "starting %s mode at %.2f seconds "
        "(%" PRIu64 " conflicts, %" PRIu64 " ticks"
#ifdef METRICS
	", %" PRIu64 " propagations, %" PRIu64 " visits"
#endif
	")", mode_string (solver),
        solver->mode.entered, solver->mode.conflicts, solver->mode.ticks
#ifdef METRICS
        , solver->mode.propagations, solver->mode.visits
#endif
	);
// clang-format on
#endif
  } else
    kissat_very_verbose (solver,
                         "no need to set mode limit (only %s mode enabled)",
                         mode_string (solver));
}

/* EVOLVE-BLOCK-START */
/* Mode-switching policy. update_mode_limit is called on every switch, after
   solver->stable has been set to the new mode; delta_ticks is the search ticks
   spent in the mode just left. It sets the limit at which the current mode
   ends: limits->mode.ticks for a stable phase (limits->mode.count odd),
   limits->mode.conflicts for a focused phase (count even).
   kissat_switching_search_mode is checked in the search loop and returns true
   when the current mode should end. Stock Kissat: a focused phase lasts
   modeint * nlogpown (n, 4) conflicts for the n-th focused phase, and each
   stable phase gets as many ticks as the focused phase before it. */
static void update_mode_limit (kissat *solver, uint64_t delta_ticks) {
  kissat_init_averages (solver, &AVERAGES);

  limits *limits = &solver->limits;
  statistics *statistics = &solver->statistics;

  assert (GET_OPTION (stable) == 1);

  if (limits->mode.count & 1) {
    limits->mode.ticks = statistics->search_ticks + delta_ticks;
    assert (solver->stable);
  } else {
    assert (limits->mode.ticks);
    const uint64_t interval = GET_OPTION (modeint);
    const uint64_t count = (statistics->switched + 1) / 2;
    const uint64_t scaled = interval * kissat_nlogpown (count, 4);
    limits->mode.conflicts = statistics->conflicts + scaled;
    assert (!solver->stable);
  }

  solver->mode.ticks = statistics->search_ticks;
  solver->mode.conflicts = statistics->conflicts;
}

bool kissat_switching_search_mode (kissat *solver) {
  assert (!solver->inconsistent);

  if (GET_OPTION (stable) != 1)
    return false;

  limits *limits = &solver->limits;
  statistics *statistics = &solver->statistics;

  if (limits->mode.count & 1)
    return statistics->search_ticks >= limits->mode.ticks;
  else
    return statistics->conflicts >= limits->mode.conflicts;
}
/* EVOLVE-BLOCK-END */

static void report_switching_from_mode (kissat *solver,
                                        uint64_t *delta_ticks) {
  statistics *statistics = &solver->statistics;
  *delta_ticks = statistics->search_ticks - solver->mode.ticks;

#ifndef QUIET
  if (kissat_verbosity (solver) < 2)
    return;

  const double current_time = kissat_process_time ();
  const double delta_time = current_time - solver->mode.entered;

  const uint64_t delta_conflicts =
      statistics->conflicts - solver->mode.conflicts;
#ifdef METRICS
  const uint64_t delta_propagations =
      statistics->search_propagations - solver->mode.propagations;
#endif
  solver->mode.entered = current_time;

  // clang-format off
  kissat_very_verbose (solver, "%s mode took %.2f seconds "
    "(%s conflicts, %s ticks"
#ifdef METRICS
    ", %s propagations"
#endif
    ")", solver->stable ? "stable" : "focused",
    delta_time, FORMAT_COUNT (delta_conflicts), FORMAT_COUNT (*delta_ticks)
#ifdef METRICS
    , FORMAT_COUNT (delta_propagations)
#endif
    );
  // clang-format on
#else
  (void) solver;
#endif
}

static void switch_to_focused_mode (kissat *solver) {
  assert (solver->stable);
  uint64_t delta;
  report_switching_from_mode (solver, &delta);
  REPORT (0, ']');
  STOP (stable);
  INC (focused_modes);
  kissat_phase (solver, "focus", GET (focused_modes),
                "switching to focused mode after %s conflicts",
                FORMAT_COUNT (CONFLICTS));
  solver->stable = false;
  update_mode_limit (solver, delta);
  START (focused);
  REPORT (0, '{');
  kissat_reset_search_of_queue (solver);
  kissat_update_focused_restart_limit (solver);
}

static void switch_to_stable_mode (kissat *solver) {
  assert (!solver->stable);
  uint64_t delta;
  report_switching_from_mode (solver, &delta);
  REPORT (0, '}');
  STOP (focused);
  INC (stable_modes);
  solver->stable = true;
  kissat_phase (solver, "stable", GET (stable_modes),
                "switched to stable mode after %" PRIu64 " conflicts",
                CONFLICTS);
  update_mode_limit (solver, delta);
  START (stable);
  REPORT (0, '[');
  kissat_init_reluctant (solver);
  kissat_update_scores (solver);
}

void kissat_switch_search_mode (kissat *solver) {
  assert (kissat_switching_search_mode (solver));

  INC (switched);
  solver->limits.mode.count++;

  if (solver->stable)
    switch_to_focused_mode (solver);
  else
    switch_to_stable_mode (solver);

  solver->averages[solver->stable].saved_decisions = DECISIONS;

  kissat_start_random_sequence (solver);
}
