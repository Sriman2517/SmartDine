# SmartDine

SmartDine is an intelligent restaurant seating optimization system that uses stochastic simulation and queueing theory to compare table assignment policies. It models time-varying customer arrivals, reservations, no-shows, service times, table capacities, and customer patience to recommend the best seating strategy for different restaurant traffic conditions.

## Features

- Non-homogeneous Poisson process arrival simulation
- Realistic lunch and dinner traffic peaks
- Multiple traffic scenarios, including Normal, Rainy Day, Dinner Rush, Super Busy, and Light Traffic
- Table capacity modeling for 2-seater, 4-seater, and 6-seater tables
- Reservation ratio and no-show rate modeling
- Seating policy comparison:
  - Best-Fit
  - Hold-Back
  - Overbooking
- Queueing theory metrics:
  - Service rate
  - Average wait time
  - Table utilization
  - Traffic intensity
  - Stability index
- Streamlit dashboard with interactive charts
- Live table assignment mode using the recommended policy
- CSV and PNG result output

## Project Structure

```text
SmartDine/
├── main.py                  # Command-line simulation runner
├── models.py                # Arrival, service, party size, and seating policy logic
├── utils.py                 # Scenario selection, scoring, metrics, and plotting
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── results/
│   └── output.csv           # Existing result output
├── teammembers.txt          # Team details
└── README.md
```

## Requirements

Install Python 3.10 or newer, then install the required packages:

```bash
pip install numpy pandas matplotlib streamlit plotly
```

## How to Run

### 1. Run the command-line simulation

From the project root:

```bash
python main.py
```

This runs predefined test scenarios and compares all policies. Results are saved to:

```text
results/all_policies_comparison.csv
results/simulation_results.png
```

### 2. Run the dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

Then open the local Streamlit URL shown in the terminal.

## Seating Policies

### Best-Fit

Assigns each party to the smallest available table that can fit the group. This works well in lighter traffic because it keeps seating simple and avoids wasting table capacity.

### Hold-Back

Protects some tables for upcoming reservations within a configurable time window. This is useful when reservation volume is high and the restaurant needs to avoid seating walk-ins at tables that may soon be needed.

### Overbooking

Allows a limited number of customers to be accepted beyond immediate physical capacity, using virtual waiting. This can improve service rate during very busy periods, but may increase wait time.

## Simulation Model

- Arrival process: non-homogeneous Poisson process
- Service time: lognormal distribution with average dining time near 70 minutes
- Party size: discrete probability distribution from 1 to 6 guests
- Customer behavior: some customers wait if no table is immediately available, while others leave
- Evaluation: weighted score based on service rate, utilization, wait time, and queue stability

## Dashboard Usage

The Streamlit dashboard lets you:

1. Select a traffic scenario.
2. Choose policies to compare.
3. Configure table counts.
4. Adjust reservation ratio, no-show rate, opening time, closing time, and hold-back window.
5. Run multiple simulation replications.
6. View charts and policy recommendations.
7. Download simulation results as CSV.
8. Enable Live Mode to test real-time table assignment.

## Outputs

The simulation reports:

- Average customers served
- Average customers turned away
- Average waiting time
- Table utilization
- Service rate
- Traffic intensity
- Stability index
- Final policy score
- Recommended policy

## Team

See `teammembers.txt` for project team details.
