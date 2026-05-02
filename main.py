import numpy as np
import pandas as pd
from models import generate_arrivals, service_time, party_size, seat_party
from utils import choose_scenario, calculate_queueing_metrics, calculate_policy_score, plot_results
import heapq
import os

# Restaurant Configuration
TABLES = {2: 10, 4: 8, 6: 4}
OPEN_TIME = 0
CLOSE_TIME = 600
RUNS = 10  # Number of simulation runs per policy

# Three test scenarios to verify all policies work
TEST_SCENARIOS = [
    {
        "name": "Rainy Day",
        "reservation_ratio": 15,
        "no_show_rate": 10,
        "expected_winner": "Best-Fit"
    },
    {
        "name": "Normal",
        "reservation_ratio": 50,
        "no_show_rate": 10,
        "expected_winner": "Hold-Back"
    },
    {
        "name": "Super Busy",
        "reservation_ratio": 25,
        "no_show_rate": 10,
        "expected_winner": "Overbooking"
    }
]

print("=" * 80)
print("SMARTDINE RESTAURANT QUEUEING SYSTEM SIMULATION")
print("=" * 80)
print(f"Tables: {TABLES}")
print(f"Runs per policy: {RUNS}")
print("=" * 80)
print("\n🧪 Running 3 test scenarios to verify all policies can win...\n")

all_test_results = []

for test_idx, test_config in enumerate(TEST_SCENARIOS, 1):
    SCENARIO = test_config["name"]
    RESERVATION_RATIO = test_config["reservation_ratio"]
    NO_SHOW_RATE = test_config["no_show_rate"]

    print(f"\n{'=' * 80}")
    print(f"TEST {test_idx}/3: {SCENARIO}")
    print(f"Reservation Ratio: {RESERVATION_RATIO}% | No-Show Rate: {NO_SHOW_RATE}%")
    print(f"Expected Winner: {test_config['expected_winner']}")
    print('=' * 80)

    lambda_func = choose_scenario(SCENARIO)
    policies = ["Best-Fit", "Hold-Back", "Overbooking"]
    scenario_results = []

    # --- Collect stats per policy across all runs ---
    policy_stats = {
        policy: {
            "served_list": [],
            "wait_list": [],
            "away_list": [],
            "util_list": [],
            "service_time_list": []
        }
        for policy in policies
    }

    # Also track arrivals per run (common to all policies)
    arrivals_per_run = []

    # === MAIN LOOP: FOR EACH RUN, SHARE ARRIVALS ACROSS ALL POLICIES ===
    for run in range(RUNS):
        # Generate arrivals ONCE for this run (shared across policies)
        arrivals = generate_arrivals(lambda_func, OPEN_TIME, CLOSE_TIME)
        arrivals = sorted(arrivals)
        arrivals_per_run.append(len(arrivals))

        # Generate reservations ONCE for this run (shared)
        num_reservations = int(len(arrivals) * RESERVATION_RATIO / 100)
        if num_reservations > 0:
            reservation_times = np.random.choice(
                arrivals, size=num_reservations, replace=False
            )
            reservations = [(t, party_size()) for t in reservation_times]
            reservations.sort()
        else:
            reservations = []
            reservation_times = []

        # Simulate no-shows ONCE for this run
        if num_reservations > 0:
            no_show_mask = np.random.choice(
                [True, False],
                size=num_reservations,
                p=[NO_SHOW_RATE / 100, 1 - NO_SHOW_RATE / 100]
            )
        else:
            no_show_mask = []

        # Base reservation status for this run
        base_reservation_status = {}
        for i in range(num_reservations):
            if no_show_mask[i]:
                base_reservation_status[reservations[i][0]] = 'no-show'
            else:
                base_reservation_status[reservations[i][0]] = 'pending'

        # Now run ALL POLICIES using the SAME arrivals/reservations
        for POLICY in policies:
            # Copy mutable structures so policies don't interfere with each other
            reservation_status = dict(base_reservation_status)
            free_tables = TABLES.copy()
            event_queue = []

            served = 0
            turned_away = 0
            wait_times = []
            total_wait_time = 0
            system_times = []
            service_times = []

            # --- SIMULATION LOOP ---
            for arrival_time in arrivals:
                # Skip if this arrival is a reservation marked as no-show
                if arrival_time in reservation_status and reservation_status[arrival_time] == 'no-show':
                    continue

                # Release finished tables
                while event_queue and event_queue[0][0] <= arrival_time:
                    end_time, cap = heapq.heappop(event_queue)
                    free_tables[cap] += 1

                party = party_size()

                # Future pending reservations
                future_reservations = [
                    r for r in reservations
                    if r[0] > arrival_time and reservation_status.get(r[0], 'pending') == 'pending'
                ]

                # Try to seat
                assigned_table = seat_party(
                    policy=POLICY,
                    free_tables=free_tables,
                    party=party,
                    current_time=arrival_time,
                    reservations=future_reservations,
                    tables_total=TABLES,
                    holdback_window=60
                )

                if assigned_table and assigned_table != "virtual":
                    # Seated at physical table
                    if arrival_time in reservation_status:
                        reservation_status[arrival_time] = 'seated'

                    free_tables[assigned_table] -= 1
                    service_duration = service_time()
                    departure_time = arrival_time + service_duration
                    heapq.heappush(event_queue, (departure_time, assigned_table))

                    served += 1
                    wait_times.append(0)
                    system_times.append(service_duration)
                    service_times.append(service_duration)

                elif assigned_table == "virtual":
                    # Virtual overbooking seat (wait + then dine)
                    if arrival_time in reservation_status:
                        reservation_status[arrival_time] = 'seated'

                    wait_duration = np.random.uniform(10, 30)
                    service_duration = service_time()
                    total_time = wait_duration + service_duration

                    served += 1
                    wait_times.append(wait_duration)
                    system_times.append(total_time)
                    total_wait_time += wait_duration
                    service_times.append(service_duration)

                else:
                    # No table right now: customer may wait or leave
                    patience_threshold = 0.6
                    if np.random.random() < patience_threshold:
                        # They agree to wait
                        if arrival_time in reservation_status:
                            reservation_status[arrival_time] = 'seated'

                        wait_duration = np.random.uniform(5, 25)
                        service_duration = service_time()
                        total_time = wait_duration + service_duration

                        served += 1
                        wait_times.append(wait_duration)
                        system_times.append(total_time)
                        total_wait_time += wait_duration
                        service_times.append(service_duration)
                    else:
                        # They leave (turned away)
                        turned_away += 1

            # --- METRICS FOR THIS RUN & POLICY ---
            total_capacity = sum(TABLES.values())
            theoretical_max = total_capacity * (CLOSE_TIME / 70)
            utilization = (served / theoretical_max) * 100 if theoretical_max > 0 else 0

            stats = policy_stats[POLICY]
            stats["served_list"].append(served)
            stats["wait_list"].append(np.mean(wait_times) if wait_times else 0)
            stats["away_list"].append(turned_away)
            stats["util_list"].append(min(utilization, 100))
            stats["service_time_list"].extend(service_times)

    # === AFTER ALL RUNS: AGGREGATE PER POLICY ===
    avg_total_arrivals = float(np.mean(arrivals_per_run)) if arrivals_per_run else 0.0

    for POLICY in policies:
        stats = policy_stats[POLICY]

        avg_served = float(np.mean(stats["served_list"])) if stats["served_list"] else 0.0
        avg_wait = float(np.mean(stats["wait_list"])) if stats["wait_list"] else 0.0
        avg_away = float(np.mean(stats["away_list"])) if stats["away_list"] else 0.0
        avg_util = float(np.mean(stats["util_list"])) if stats["util_list"] else 0.0
        avg_service = float(np.mean(stats["service_time_list"])) if stats["service_time_list"] else 70.0

        # Service rate based on served vs arrivals (served + away)
        service_rate = (avg_served / (avg_served + avg_away)) * 100 if (avg_served + avg_away) > 0 else 0.0

        # Queueing metrics use the *average* arrivals per run
        queueing_metrics = calculate_queueing_metrics(
            avg_total_arrivals, avg_served, avg_wait, avg_service, CLOSE_TIME
        )

        score = calculate_policy_score(
            POLICY,
            {
                'service_rate': service_rate,
                'utilization': avg_util,
                'avg_wait': avg_wait
            },
            queueing_metrics,
            SCENARIO,
            RESERVATION_RATIO
        )

        scenario_results.append({
            'Scenario': SCENARIO,
            'Policy': POLICY,
            'Avg_Served': avg_served,
            'Avg_Turned_Away': avg_away,
            'Service_Rate_%': service_rate,
            'Avg_Wait_Min': avg_wait,
            'Utilization_%': avg_util,
            'Traffic_Intensity_rho': queueing_metrics['rho'],
            'Stability_Index': queueing_metrics['stability_index'],
            'Score': score
        })

        print(f"\n  📊 {POLICY} policy:")
        print(f"    ✅ Score = {score:.4f}, Service Rate = {service_rate:.1f}%, "
              f"Avg Served = {avg_served:.2f}, Avg Turned Away = {avg_away:.2f}, "
              f"Avg Wait = {avg_wait:.2f} min, Utilization = {avg_util:.2f}%")

    # Create results DataFrame for this scenario
    scenario_df = pd.DataFrame(scenario_results)

    # Determine best policy for this scenario
    best_policy_row = scenario_df.loc[scenario_df['Score'].idxmax()]
    actual_winner = best_policy_row['Policy']

    print(f"\n  {'=' * 76}")
    print(f"  🏆 WINNER FOR {SCENARIO}: {actual_winner}")
    print(f"  {'=' * 76}")

    if actual_winner == test_config['expected_winner']:
        print(f"  ✅ PASS: {actual_winner} won as expected!")
    else:
        print(f"  ⚠️  NOTE: {actual_winner} won (expected {test_config['expected_winner']})")

    print(f"\n  Detailed Scores:")
    for _, row in scenario_df.iterrows():
        marker = "👑" if row['Policy'] == actual_winner else "  "
        print(f"    {marker} {row['Policy']:12s}: {row['Score']:.4f} | "
              f"Service={row['Service_Rate_%']:5.1f}% | "
              f"Wait={row['Avg_Wait_Min']:5.2f}min | "
              f"Util={row['Utilization_%']:5.1f}%")

    all_test_results.extend(scenario_results)

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY - ALL SCENARIOS")
print("=" * 80)

final_df = pd.DataFrame(all_test_results)
print(final_df.to_string(index=False))

print("\n" + "=" * 80)
print("POLICY WINS BY SCENARIO")
print("=" * 80)

for scenario_name in [s["name"] for s in TEST_SCENARIOS]:
    scenario_data = final_df[final_df['Scenario'] == scenario_name]
    winner = scenario_data.loc[scenario_data['Score'].idxmax(), 'Policy']
    print(f"  {scenario_name:15s}: 🏆 {winner}")

print("=" * 80)

# Save results
os.makedirs("results", exist_ok=True)
final_df.to_csv("results/all_policies_comparison.csv", index=False)
print("\n✅ Results saved to results/all_policies_comparison.csv")

# Visualization for best overall policy
avg_scores = final_df.groupby('Policy')['Score'].mean()
best_overall = avg_scores.idxmax()

print(f"\n🌟 BEST OVERALL POLICY (across all scenarios): {best_overall}")
print(f"   Average Score: {avg_scores[best_overall]:.4f}")

best_df = pd.DataFrame({
    "Policy": [best_overall],
    "Total_Arrivals": [int(
        final_df[final_df['Policy'] == best_overall]['Avg_Served'].mean() +
        final_df[final_df['Policy'] == best_overall]['Avg_Turned_Away'].mean()
    )],
    "Served": [int(final_df[final_df['Policy'] == best_overall]['Avg_Served'].mean())],
    "Turned_Away": [int(final_df[final_df['Policy'] == best_overall]['Avg_Turned_Away'].mean())],
    "Service_Rate_%": [final_df[final_df['Policy'] == best_overall]['Service_Rate_%'].mean()],
    "Avg_Wait_Min": [final_df[final_df['Policy'] == best_overall]['Avg_Wait_Min'].mean()],
    "Max_Wait_Min": [final_df[final_df['Policy'] == best_overall]['Avg_Wait_Min'].max()],
    "Avg_System_Time_Min": [final_df[final_df['Policy'] == best_overall]['Avg_Wait_Min'].mean() + 70],
    "Table_Utilization_%": [final_df[final_df['Policy'] == best_overall]['Utilization_%'].mean()],
    "Total_Wait_Time_Min": [
        final_df[final_df['Policy'] == best_overall]['Avg_Wait_Min'].mean() *
        final_df[final_df['Policy'] == best_overall]['Avg_Served'].mean()
    ]
})

plot_results(best_df)
