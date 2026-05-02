import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from models import generate_arrivals, service_time, party_size, seat_party
from utils import choose_scenario, calculate_queueing_metrics, calculate_policy_score
import heapq
import time

st.set_page_config(page_title="SmartDine – Restaurant Simulation", layout="wide")
st.title("🍽️ SmartDine - An intelligent restaurant seating optimization system using stochastic processes.")
st.markdown("### Non-Homogeneous Poisson Process with Queueing Theory Analysis")

# ---------------- Sidebar Configuration ----------------
st.sidebar.header("⚙️ Simulation Settings")

scenario = st.sidebar.selectbox("Traffic Scenario",
                                ["Normal", "Dinner Rush", "Rainy Day", "Super Busy", "Light Traffic"])
policies = ["Best-Fit", "Hold-Back", "Overbooking"]
selected_policies = st.sidebar.multiselect("Select Policies to Compare",
                                           policies, default=policies)

st.sidebar.markdown("### 🪑 Table Configuration")
tables_2 = st.sidebar.number_input("2-Seater Tables", 5, 20, 10)
tables_4 = st.sidebar.number_input("4-Seater Tables", 5, 20, 8)
tables_6 = st.sidebar.number_input("6-Seater Tables", 2, 10, 4)

# ---------------- Advanced Settings ----------------
with st.sidebar.expander("🔧 Advanced Settings"):
    open_time = st.number_input("Opening Time (minutes)", 0, 1440, 0)
    close_time = st.number_input("Closing Time (minutes)", 60, 1440, 600)
    reservation_ratio = st.slider("Reservation Ratio (%)", 0, 100, 30)
    no_show_rate = st.slider("No-Show Rate (% of reservations)", 0, 50, 10)
    holdback_window = st.slider("Hold-Back Window (minutes)", 30, 120, 60)

runs = st.sidebar.slider("Number of Replications", 1, 50, 10)
run_btn = st.sidebar.button("🚀 Run Simulations", type="primary")

# ---------------- Information Display ----------------
st.sidebar.markdown("---")
st.sidebar.info("""
**Policy Descriptions:**
- **Best-Fit**: Seats in smallest available table
- **Hold-Back**: Reserves tables for upcoming reservations
- **Overbooking**: Accepts customers beyond capacity (dynamic)
""")

# ---------------- Real-Time Table Assignment Section ----------------
st.sidebar.markdown("---")
st.sidebar.header("🔴 LIVE Table Assignment")
enable_live = st.sidebar.checkbox("Enable Live Mode", value=False)

if enable_live and 'recommendation' in st.session_state:
    policy = st.session_state['recommendation']['policy']
    st.sidebar.success(f"Using: **{policy}**")
    
    # Warning for Hold-Back policy
    if policy == 'Hold-Back':
        st.sidebar.info("📅 Hold-Back policy activated! Add reservations below to see it in action.")
    
    # RESERVATION MANAGEMENT (NEW!)
    with st.sidebar.expander("📅 Manage Reservations", expanded=(policy == 'Hold-Back')):
        st.markdown("#### Add Reservation")
        
        with st.form("add_reservation_form"):
            res_time = st.number_input("Time from now (min)", 5, 120, 30, key="res_time", 
                                      help="How many minutes from now?")
            res_party = st.number_input("Party Size", 1, 6, 4, key="res_party")
            submit_res = st.form_submit_button("➕ Add Reservation")
        
        if submit_res:
            # Initialize reservations list if needed
            if 'live_reservations' not in st.session_state:
                st.session_state['live_reservations'] = []
            if 'current_time' not in st.session_state:
                st.session_state['current_time'] = 0
            
            future_time = st.session_state['current_time'] + res_time
            st.session_state['live_reservations'].append({
                'time': future_time,
                'party': res_party,
                'status': 'pending',
                'id': len(st.session_state['live_reservations'])
            })
            st.success(f"✅ Reserved for {res_party} people at +{res_time}min")
        
        # Display current reservations
        if 'live_reservations' in st.session_state and st.session_state['live_reservations']:
            st.markdown("**Upcoming Reservations:**")
            current_time = st.session_state.get('current_time', 0)
            
            pending_reservations = [r for r in st.session_state['live_reservations'] 
                                   if r['status'] == 'pending']
            
            if pending_reservations:
                for res in pending_reservations:
                    time_until = res['time'] - current_time
                    if time_until > 0:
                        st.info(f"🕐 Party of {res['party']} in {time_until:.0f}min")
                    elif time_until >= -5:  # Just passed
                        st.warning(f"⏰ Party of {res['party']} NOW!")
                    else:  # Old reservation
                        res['status'] = 'expired'
            else:
                st.write("No pending reservations")
            
            # Clear old reservations button
            if st.button("🗑️ Clear All Reservations"):
                st.session_state['live_reservations'] = []
                st.rerun()
    
    # TABLE ASSIGNMENT FORM
    with st.sidebar.form("assign_table_form"):
        st.markdown("#### Assign New Party (Walk-in)")
        live_party_size = st.number_input("Party Size", 1, 6, 2, key="live_party")
        live_duration = st.number_input("Duration (min)", 30, 120, 70, key="live_duration")
        is_reservation = st.checkbox("This is a reservation arrival", value=False)
        submit_assign = st.form_submit_button("✅ Assign Table")
        
    if submit_assign:
        # Initialize live tables if needed
        if 'live_tables' not in st.session_state:
            st.session_state['live_tables'] = {}
            for i in range(1, tables_2 + 1):
                st.session_state['live_tables'][f'2-{i}'] = {
                    'capacity': 2, 'occupied': False, 'end_time': None, 'party': None
                }
            for i in range(1, tables_4 + 1):
                st.session_state['live_tables'][f'4-{i}'] = {
                    'capacity': 4, 'occupied': False, 'end_time': None, 'party': None
                }
            for i in range(1, tables_6 + 1):
                st.session_state['live_tables'][f'6-{i}'] = {
                    'capacity': 6, 'occupied': False, 'end_time': None, 'party': None
                }
            st.session_state['current_time'] = 0
            st.session_state['assignment_history'] = []
        
        if 'live_reservations' not in st.session_state:
            st.session_state['live_reservations'] = []
        
        # Calculate free tables
        free_tables = {2: 0, 4: 0, 6: 0}
        for table_id, table_info in st.session_state['live_tables'].items():
            if not table_info['occupied']:
                free_tables[table_info['capacity']] += 1
        
        TABLES = {2: tables_2, 4: tables_4, 6: tables_6}
        current_time = st.session_state['current_time']
        
        # Format reservations for seat_party function
        active_reservations = [
            (r['time'], r['party']) 
            for r in st.session_state['live_reservations'] 
            if r['status'] == 'pending' and r['time'] > current_time
        ]
        
        # Try to assign table
        assigned = seat_party(
            policy=policy,
            free_tables=free_tables,
            party=live_party_size,
            current_time=current_time,
            reservations=active_reservations,  # ✅ NOW HAS RESERVATION DATA!
            tables_total=TABLES,
            holdback_window=holdback_window
        )
        
        if assigned and assigned != "virtual":
            # Find first available table of that capacity
            for table_id, table_info in st.session_state['live_tables'].items():
                if table_info['capacity'] == assigned and not table_info['occupied']:
                    st.session_state['live_tables'][table_id] = {
                        'capacity': assigned,
                        'occupied': True,
                        'end_time': current_time + live_duration,
                        'party': live_party_size
                    }
                    st.session_state['assignment_history'].append({
                        'time': current_time,
                        'table': table_id,
                        'party': live_party_size,
                        'duration': live_duration,
                        'policy': policy,
                        'type': 'Reservation' if is_reservation else 'Walk-in'
                    })
                    
                    # If this was a reservation arrival, mark it as seated
                    if is_reservation and active_reservations:
                        # Mark the closest reservation as seated
                        for res in st.session_state['live_reservations']:
                            if (res['status'] == 'pending' and 
                                abs(res['time'] - current_time) < 5 and 
                                res['party'] == live_party_size):
                                res['status'] = 'seated'
                                break
                    
                    st.sidebar.success(f"✅ Assigned **{table_id}** for party of {live_party_size}!")
                    break
        elif assigned == "virtual":
            st.session_state['assignment_history'].append({
                'time': current_time,
                'table': 'VIRTUAL',
                'party': live_party_size,
                'duration': live_duration,
                'policy': policy,
                'type': 'Walk-in (Waiting)'
            })
            st.sidebar.warning(f"⚠️ Overbooked! Party of {live_party_size} accepted but will wait (~15 min).")
        else:
            st.sidebar.error(f"❌ No table available for party of {live_party_size}")
            
            # Show why Hold-Back rejected if applicable
            if policy == 'Hold-Back' and active_reservations:
                st.sidebar.info(f"ℹ️ Hold-Back protecting tables for {len(active_reservations)} upcoming reservations")

elif enable_live:
    st.sidebar.warning("⚠️ Run simulation first to get recommended policy!")

# ---------------- Simulation ----------------
if run_btn:
    if not selected_policies:
        st.error("⚠️ Please select at least one policy to compare!")
    else:
        results_all = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        lambda_func = choose_scenario(scenario)
        TABLES = {2: tables_2, 4: tables_4, 6: tables_6}
        
        # STEP 1: Generate all arrivals and reservations for ALL runs (ONCE)
        all_runs_data = []
        
        for run in range(runs):
            # Generate arrivals ONCE per run
            arrivals = generate_arrivals(lambda_func, open_time, close_time)
            arrivals = sorted(arrivals)
            
            # Generate reservations ONCE per run
            num_res = int(len(arrivals) * reservation_ratio / 100)
            if num_res > 0:
                res_indices = np.random.choice(len(arrivals), size=num_res, replace=False)
                res_times = [arrivals[i] for i in res_indices]
                reservations = [(t, party_size()) for t in res_times]
                reservations.sort()
                
                no_show_mask = np.random.choice(
                    [True, False],
                    size=num_res,
                    p=[no_show_rate/100, 1 - no_show_rate/100]
                )
                
                base_reservation_status = {}
                for i in range(num_res):
                    if no_show_mask[i]:
                        base_reservation_status[reservations[i][0]] = 'no-show'
                    else:
                        base_reservation_status[reservations[i][0]] = 'pending'
            else:
                reservations = []
                base_reservation_status = {}
            
            all_runs_data.append({
                'arrivals': arrivals,
                'reservations': reservations,
                'base_reservation_status': base_reservation_status
            })
        
        # STEP 2: Run each policy using the SAME arrivals/reservations
        total_sims = len(selected_policies) * runs
        current_sim = 0
        
        for policy_idx, policy in enumerate(selected_policies):
            served_list, wait_list, away_list, util_list = [], [], [], []
            service_time_list = []

            for run in range(runs):
                current_sim += 1
                progress_bar.progress(current_sim / total_sims)
                status_text.text(f"Running {policy} - Replication {run+1}/{runs}")
                
                # Use the SAME arrivals and reservations as other policies
                run_data = all_runs_data[run]
                arrivals = run_data['arrivals']
                reservations = run_data['reservations']
                reservation_status = dict(run_data['base_reservation_status'])
                
                served, turned_away = 0, 0
                wait_times = []
                event_queue = []
                free_tables = TABLES.copy()
                service_times = []

                for a in arrivals:
                    if a in reservation_status and reservation_status[a] == 'no-show':
                        continue

                    while event_queue and event_queue[0][0] <= a:
                        end_time, cap = heapq.heappop(event_queue)
                        free_tables[cap] += 1

                    p = party_size()
                    
                    future_reservations = [
                        r for r in reservations 
                        if r[0] > a and reservation_status.get(r[0], 'pending') == 'pending'
                    ]
                    
                    choice = seat_party(
                        policy, free_tables, p, a, future_reservations,
                        tables_total=TABLES, holdback_window=holdback_window
                    )

                    if choice and choice != "virtual":
                        if a in reservation_status:
                            reservation_status[a] = 'seated'
                        
                        free_tables[choice] -= 1
                        svc_time = service_time()
                        heapq.heappush(event_queue, (a + svc_time, choice))
                        served += 1
                        wait_times.append(0)
                        service_times.append(svc_time)
                        
                    elif choice == "virtual":
                        if a in reservation_status:
                            reservation_status[a] = 'seated'
                        
                        wait_time = np.random.uniform(10, 20)
                        served += 1
                        wait_times.append(wait_time)
                        service_times.append(service_time())
                        
                    else:
                        if np.random.random() < 0.6:
                            if a in reservation_status:
                                reservation_status[a] = 'seated'
                            
                            wait_time = np.random.uniform(5, 25)
                            wait_times.append(wait_time)
                            served += 1
                            service_times.append(service_time())
                        else:
                            turned_away += 1

                total_capacity = sum(TABLES.values())
                theoretical_max = total_capacity * ((close_time - open_time) / 70)
                utilization = (served / theoretical_max) * 100 if theoretical_max > 0 else 0
                
                served_list.append(served)
                wait_list.append(np.mean(wait_times) if wait_times else 0)
                away_list.append(turned_away)
                util_list.append(min(utilization, 100))
                service_time_list.extend(service_times)

            # Calculate aggregate metrics
            avg_served = np.mean(served_list)
            avg_wait = np.mean(wait_list)
            avg_away = np.mean(away_list)
            avg_util = np.mean(util_list)
            avg_service = np.mean(service_time_list) if service_time_list else 70
            service_rate = (avg_served / (avg_served + avg_away)) * 100 if (avg_served + avg_away) > 0 else 0
            
            # Average number of arrivals per run
            avg_arrivals = np.mean([len(rd['arrivals']) for rd in all_runs_data])
            
            # Queueing metrics
            queueing_metrics = calculate_queueing_metrics(
                avg_arrivals, avg_served, avg_wait, avg_service, close_time
            )
            
            # Calculate score
            score = calculate_policy_score(
                policy,
                {'service_rate': service_rate, 'utilization': avg_util, 'avg_wait': avg_wait},
                queueing_metrics,
                scenario,
                reservation_ratio
            )
            
            df = pd.DataFrame({
                "Scenario": [scenario],
                "Policy": [policy],
                "Avg Served": [avg_served],
                "Std Served": [np.std(served_list)],
                "Avg Turned Away": [avg_away],
                "Std Turned Away": [np.std(away_list)],
                "Avg Wait (min)": [avg_wait],
                "Std Wait": [np.std(wait_list)],
                "Utilization (%)": [avg_util],
                "Service Rate (%)": [service_rate],
                "Traffic Intensity (ρ)": [queueing_metrics['rho']],
                "Stability Index": [queueing_metrics['stability_index']],
                "Score": [score]
            })
            results_all.append(df)

        progress_bar.empty()
        status_text.empty()

        final_df = pd.concat(results_all, ignore_index=True)
        st.success(f"✅ Completed {len(selected_policies)} policies × {runs} replications = {total_sims} simulations")

        # Find best policy based on score
        best_idx = final_df['Score'].idxmax()
        best_policy_row = final_df.iloc[best_idx]
        
        # Store recommendation in session state
        st.session_state['recommendation'] = {
            'policy': best_policy_row['Policy'],
            'score': best_policy_row['Score'],
            'service_rate': best_policy_row['Service Rate (%)'],
            'avg_wait': best_policy_row['Avg Wait (min)'],
            'utilization': best_policy_row['Utilization (%)'],
            'rho': best_policy_row['Traffic Intensity (ρ)'],
            'stability': best_policy_row['Stability Index']
        }

        st.subheader(f"📊 Results for Scenario: {scenario}")
        
        display_df = final_df.copy()
        display_df = display_df.round(2)
        st.dataframe(display_df, use_container_width=True, height=200)

        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv,
            file_name=f"SmartDine_{scenario}_Results.csv",
            mime="text/csv"
        )

        # ---------------- Visualizations ----------------
        st.markdown("---")
        st.subheader("📈 Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                final_df, 
                x="Policy", 
                y="Avg Wait (min)", 
                color="Policy",
                title="Average Waiting Time",
                error_y="Std Wait",
                labels={"Avg Wait (min)": "Wait Time (minutes)"}
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.bar(
                final_df, 
                x="Policy", 
                y="Utilization (%)", 
                color="Policy",
                title="Table Utilization",
                labels={"Utilization (%)": "Utilization (%)"}
            )
            fig2.update_layout(showlegend=False, yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        
        with col3:
            fig3 = px.bar(
                final_df,
                x="Policy",
                y="Service Rate (%)",
                color="Policy",
                title="Service Rate (% Served)",
                labels={"Service Rate (%)": "Service Rate (%)"}
            )
            fig3.update_layout(showlegend=False, yaxis_range=[0, 100])
            st.plotly_chart(fig3, use_container_width=True)
            
        with col4:
            fig5 = px.bar(
                final_df,
                x="Policy",
                y="Score",
                color="Policy",
                title="Overall Score (Queueing Theory Based)",
                labels={"Score": "Composite Score"}
            )
            fig5.update_layout(showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

        # ---------------- Smart Recommendation ----------------
        st.markdown("---")
        st.subheader("🏆 Policy Recommendation")
        
        col_rec1, col_rec2, col_rec3, col_rec4 = st.columns(4)
        
        with col_rec1:
            st.metric(
                "Recommended Policy",
                best_policy_row['Policy'],
                f"Score: {best_policy_row['Score']:.3f}"
            )
        
        with col_rec2:
            st.metric(
                "Service Rate",
                f"{best_policy_row['Service Rate (%)']:.1f}%",
                f"{best_policy_row['Avg Served']:.0f} served"
            )
        
        with col_rec3:
            st.metric(
                "Avg Wait Time",
                f"{best_policy_row['Avg Wait (min)']:.1f} min",
                f"Util: {best_policy_row['Utilization (%)']:.1f}%"
            )
        
        with col_rec4:
            st.metric(
                "Traffic Intensity",
                f"ρ = {best_policy_row['Traffic Intensity (ρ)']:.3f}",
                f"Stability: {best_policy_row['Stability Index']:.3f}"
            )
        
        st.info(f"""
        **Why {best_policy_row['Policy']}?** Based on **{scenario.lower()}** traffic with {reservation_ratio}% reservations, 
        {best_policy_row['Policy']} provides the best balance of service rate ({best_policy_row['Service Rate (%)']:.1f}%), 
        table utilization ({best_policy_row['Utilization (%)']:.1f}%), and customer wait times 
        ({best_policy_row['Avg Wait (min)']:.1f} min). 
        
        **Queueing Theory Analysis:** Traffic intensity ρ = {best_policy_row['Traffic Intensity (ρ)']:.3f} 
        (optimal < 0.85), Stability Index = {best_policy_row['Stability Index']:.3f} 
        (higher is better, indicates system won't collapse).
        """)

        with st.expander("🔍 Queueing Theory Insights"):
            st.markdown(f"""
            **Little's Law:** L = λW
            - Average customers in system is proportional to arrival rate × wait time
            
            **Traffic Intensity (ρ = λ/μ):**
            - ρ < 0.5: Light load, excellent service
            - 0.5 ≤ ρ < 0.85: Moderate load, good service
            - ρ ≥ 0.85: Heavy load, wait times increase rapidly
            - Your system: ρ = {best_policy_row['Traffic Intensity (ρ)']:.3f}
            
            **Stability Index (1 - ρ):**
            - Closer to 1.0: Very stable system
            - Closer to 0.0: System approaching saturation
            - Your system: {best_policy_row['Stability Index']:.3f}
            """)

# Live table visualization
if enable_live and 'live_tables' in st.session_state:
    st.markdown("---")
    st.subheader("🔴 LIVE Table Status")
    
    st.session_state['current_time'] += 1
    
    # Auto-release tables
    for table_id in st.session_state['live_tables']:
        table = st.session_state['live_tables'][table_id]
        if table['occupied'] and table['end_time'] is not None:
            if st.session_state['current_time'] >= table['end_time']:
                st.session_state['live_tables'][table_id]['occupied'] = False
                st.session_state['live_tables'][table_id]['party'] = None
                st.session_state['live_tables'][table_id]['end_time'] = None
    
    col_2seat, col_4seat, col_6seat = st.columns(3)
    
    with col_2seat:
        st.markdown("#### 2-Seater Tables")
        for table_id, table in st.session_state['live_tables'].items():
            if table['capacity'] == 2:
                if table['occupied']:
                    time_left = max(0, table['end_time'] - st.session_state['current_time'])
                    st.error(f"🔴 {table_id}: Party of {table['party']} (Free in {time_left:.0f}m)")
                else:
                    st.success(f"🟢 {table_id}: Available")
    
    with col_4seat:
        st.markdown("#### 4-Seater Tables")
        for table_id, table in st.session_state['live_tables'].items():
            if table['capacity'] == 4:
                if table['occupied']:
                    time_left = max(0, table['end_time'] - st.session_state['current_time'])
                    st.error(f"🔴 {table_id}: Party of {table['party']} (Free in {time_left:.0f}m)")
                else:
                    st.success(f"🟢 {table_id}: Available")
    
    with col_6seat:
        st.markdown("#### 6-Seater Tables")
        for table_id, table in st.session_state['live_tables'].items():
            if table['capacity'] == 6:
                if table['occupied']:
                    time_left = max(0, table['end_time'] - st.session_state['current_time'])
                    st.error(f"🔴 {table_id}: Party of {table['party']} (Free in {time_left:.0f}m)")
                else:
                    st.success(f"🟢 {table_id}: Available")
    
    # Assignment history
    if st.session_state.get('assignment_history'):
        st.markdown("#### Recent Assignments")
        history_df = pd.DataFrame(st.session_state['assignment_history'][-10:])
        st.dataframe(history_df, use_container_width=True)
    
    time.sleep(1)
    st.rerun()

else:
    st.info("👈 Configure your simulation parameters in the sidebar and click '🚀 Run Simulations' to begin!")
    
    st.markdown("""
    ### About This Simulation
    
    This dashboard implements a **Non-Homogeneous Poisson Process** queuing model for restaurant seating optimization.
    
    **Key Features:**
    - Time-varying arrival rates (lunch and dinner rushes)
    - Multiple table sizes with capacity constraints
    - Three seating policies: Best-Fit, Hold-Back, and Overbooking
    - Reservation management with no-show modeling
    - Comprehensive queueing theory metrics (ρ, stability index, Little's Law)
    - **Real-time table assignment** using recommended policy
    - **Live reservation management** for Hold-Back policy testing
    
    **Theoretical Foundation:**
    - Arrival process: Non-homogeneous Poisson with λ(t)
    - Service time: Lognormal distribution (mean ≈ 70 min)
    - Party size: Discrete distribution (1-6 people)
    - Queue discipline: FCFS with patience modeling
    - Scoring: Weighted composite using queueing metrics
    
    **How to Use LIVE Mode:**
    1. Run simulations to get policy recommendation
    2. Enable LIVE Mode checkbox
    3. For Hold-Back: Add reservations using the form
    4. Assign walk-in parties and watch real-time table management
    """)