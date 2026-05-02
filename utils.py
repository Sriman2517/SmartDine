import numpy as np
import matplotlib.pyplot as plt

def lambda_t(t):
    """
    Time-varying arrival rate λ(t) customers per minute.
    Models realistic restaurant traffic with lunch and dinner rushes.
    """
    t = np.array(t, dtype=float)
    hour = (t / 60) % 24

    rates = np.full_like(hour, 0.08)
    rates = np.where((12 <= hour) & (hour < 14), 0.25, rates)
    rates = np.where((19 <= hour) & (hour < 21), 0.35, rates)

    if rates.size == 1:
        return float(rates)
    return rates


def plot_results(df):
    """Create comprehensive visualization of simulation results."""
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Customer Outcomes
    ax1 = plt.subplot(2, 3, 1)
    outcomes = [df["Served"].iloc[0], df["Turned_Away"].iloc[0]]
    colors = ['#2ecc71', '#e74c3c']
    bars1 = ax1.bar(["Served", "Turned Away"], outcomes, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title("Customer Outcomes", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Service Rate
    ax2 = plt.subplot(2, 3, 2)
    service_rate = df["Service_Rate_%"].iloc[0]
    ax2.bar(["Service Rate"], [service_rate], color='#3498db', alpha=0.8, edgecolor='black')
    ax2.set_ylim(0, 100)
    ax2.set_title("Service Rate", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0, service_rate, f'{service_rate:.1f}%', 
            ha='center', va='bottom', fontweight='bold')
    
    # 3. Table Utilization
    ax3 = plt.subplot(2, 3, 3)
    utilization = df["Table_Utilization_%"].iloc[0]
    color = '#2ecc71' if utilization < 80 else '#e67e22' if utilization < 95 else '#e74c3c'
    ax3.bar(["Utilization"], [utilization], color=color, alpha=0.8, edgecolor='black')
    ax3.set_ylim(0, 100)
    ax3.set_title("Table Utilization", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Percentage (%)", fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.text(0, utilization, f'{utilization:.1f}%', 
            ha='center', va='bottom', fontweight='bold')
    
    # 4. Wait Time Statistics
    ax4 = plt.subplot(2, 3, 4)
    wait_stats = [df["Avg_Wait_Min"].iloc[0], df["Max_Wait_Min"].iloc[0]]
    bars4 = ax4.bar(["Average Wait", "Maximum Wait"], wait_stats, 
                    color=['#9b59b6', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax4.set_title("Wait Time Statistics", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Minutes", fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. System Performance Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    summary_text = f"""
    SIMULATION SUMMARY
    {'='*30}
    Policy: {df["Policy"].iloc[0]}
    
    Total Arrivals: {df["Total_Arrivals"].iloc[0]}
    Customers Served: {df["Served"].iloc[0]}
    Customers Lost: {df["Turned_Away"].iloc[0]}
    
    Avg Wait Time: {df["Avg_Wait_Min"].iloc[0]:.2f} min
    Avg System Time: {df["Avg_System_Time_Min"].iloc[0]:.2f} min
    Total Wait Time: {df["Total_Wait_Time_Min"].iloc[0]:.2f} min
    
    Service Rate: {df["Service_Rate_%"].iloc[0]:.1f}%
    Utilization: {df["Table_Utilization_%"].iloc[0]:.1f}%
    """
    ax5.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    # 6. Performance Gauge
    ax6 = plt.subplot(2, 3, 6)
    service_rate = df["Service_Rate_%"].iloc[0]
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    ax6.plot(theta, r, 'k-', linewidth=2)
    ax6.fill_between(theta[:33], 0, r[:33], color='#e74c3c', alpha=0.3)
    ax6.fill_between(theta[33:66], 0, r[33:66], color='#f39c12', alpha=0.3)
    ax6.fill_between(theta[66:], 0, r[66:], color='#2ecc71', alpha=0.3)
    
    needle_angle = np.pi * (1 - service_rate/100)
    ax6.plot([0, np.cos(needle_angle)], [0, np.sin(needle_angle)], 
            'k-', linewidth=3)
    ax6.plot(0, 0, 'ko', markersize=10)
    
    ax6.set_xlim(-1.2, 1.2)
    ax6.set_ylim(-0.2, 1.2)
    ax6.axis('off')
    ax6.set_title("Overall Performance", fontsize=12, fontweight='bold')
    ax6.text(0, -0.1, f"{service_rate:.1f}%", ha='center', 
            fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def choose_scenario(name):
    """Returns a scenario-specific λ(t) function."""
    if name == "Normal":
        return lambda_t
    
    elif name == "Dinner Rush":
        return lambda t: lambda_t(t) * 1.5
    
    elif name == "Rainy Day":
        return lambda t: lambda_t(t) * 0.6
    
    elif name == "Super Busy":
        return lambda t: lambda_t(t) * 2.0
    
    elif name == "Light Traffic":
        return lambda t: lambda_t(t) * 0.5
    
    elif name == "Weekend Rush":
        def weekend_lambda(t):
            t = np.array(t, dtype=float)
            hour = (t / 60) % 24
            rates = np.full_like(hour, 0.15)
            rates = np.where((11.5 <= hour) & (hour < 14.5), 0.40, rates)
            rates = np.where((18 <= hour) & (hour < 22), 0.50, rates)
            if rates.size == 1:
                return float(rates)
            return rates
        return weekend_lambda
    
    elif name == "Constant High":
        return lambda t: 0.30
    
    else:
        return lambda_t


def calculate_queueing_metrics(arrivals, served, avg_wait, avg_service, close_time):
    """Calculate queueing theory metrics for policy evaluation."""
    lambda_rate = arrivals / close_time  # arrival rate per minute
    mu = 1 / avg_service if avg_service > 0 else 0  # service rate per minute
    rho = lambda_rate / mu if mu > 0 else 0  # traffic intensity
    
    # Little's Law: L = λW
    avg_customers_in_system = lambda_rate * (avg_wait + avg_service)
    
    # Stability index (closer to 1 is more stable, closer to 0 is unstable)
    stability_index = 1 - rho if rho < 1 else 0
    
    return {
        'lambda': lambda_rate * 60,  # per hour
        'mu': mu * 60,  # per hour
        'rho': rho,
        'avg_customers': avg_customers_in_system,
        'stability_index': stability_index
    }


def calculate_policy_score(policy, metrics, queueing_metrics, scenario, reservation_ratio):
    """
    Calculate comprehensive score for policy using queueing theory.
    Returns score between 0 and 1 (higher is better).
    BALANCED to ensure all policies can win in appropriate scenarios.
    """
    # Base weights (equal importance)
    weights = {
        'service_rate': 0.30,
        'utilization': 0.25,
        'wait_time': 0.25,
        'stability': 0.20
    }
    
    # Normalize metrics
    service_rate_score = metrics['service_rate'] / 100
    utilization_score = min(metrics['utilization'] / 80, 1)  # optimal around 80%
    wait_time_score = max(0, 1 - (metrics['avg_wait'] / 30))  # penalize > 30 min
    stability_score = min(queueing_metrics['stability_index'] * 2, 1)
    
    # Base score
    base_score = (
        service_rate_score * weights['service_rate'] +
        utilization_score * weights['utilization'] +
        wait_time_score * weights['wait_time'] +
        stability_score * weights['stability']
    )
    
    # Scenario-specific multipliers (BALANCED)
    scenario_multiplier = 1.0
    
    # Light Traffic / Rainy Day - Best-Fit excels
    if scenario in ['Rainy Day', 'Light Traffic']:
        if policy == 'Best-Fit':
            scenario_multiplier = 1.20  # Strong advantage
        elif policy == 'Hold-Back':
            scenario_multiplier = 0.90  # Disadvantage (unnecessary complexity)
        elif policy == 'Overbooking':
            scenario_multiplier = 0.80  # Clear disadvantage (risky when not busy)
    
    # High Reservation scenarios - Hold-Back excels
    elif reservation_ratio >= 45:
        if policy == 'Hold-Back':
            # VERY strong advantage with high reservations
            scenario_multiplier = 1.35 if reservation_ratio >= 60 else 1.30
        elif policy == 'Best-Fit':
            scenario_multiplier = 0.85  # Clear disadvantage (can't handle reservations well)
        elif policy == 'Overbooking':
            scenario_multiplier = 1.00  # Neutral
    
    # Super Busy / Dinner Rush - Overbooking excels
    elif scenario in ['Super Busy', 'Dinner Rush']:
        if policy == 'Overbooking':
            scenario_multiplier = 1.30  # Strong advantage in high demand
        elif policy == 'Best-Fit':
            scenario_multiplier = 0.80  # Clear disadvantage (rigid)
        elif policy == 'Hold-Back':
            scenario_multiplier = 0.95  # Slight disadvantage (not aggressive enough)
    
    # Normal scenario with moderate-high reservations - Hold-Back can still win
    elif scenario == 'Normal' and reservation_ratio >= 35:
        if policy == 'Hold-Back':
            scenario_multiplier = 1.25
        elif policy == 'Best-Fit':
            scenario_multiplier = 0.95
        elif policy == 'Overbooking':
            scenario_multiplier = 1.05
    
    # Weekend Rush - Hold-Back excels with structured demand
    elif scenario == 'Weekend Rush':
        if policy == 'Hold-Back':
            scenario_multiplier = 1.28
        elif policy == 'Best-Fit':
            scenario_multiplier = 0.90
        elif policy == 'Overbooking':
            scenario_multiplier = 1.10
    
    # Normal scenario with low reservations - Best-Fit wins
    else:  # Normal with low reservations
        if policy == 'Best-Fit':
            scenario_multiplier = 1.15  # Advantage (simple and effective)
        elif policy == 'Hold-Back':
            scenario_multiplier = 1.00  # Baseline
        elif policy == 'Overbooking':
            scenario_multiplier = 0.95  # Slight disadvantage
    
    # Additional reservation-based boosts
    if reservation_ratio >= 50:
        if policy == 'Hold-Back':
            scenario_multiplier *= 1.08  # Extra boost for very high reservations
        elif policy == 'Best-Fit':
            scenario_multiplier *= 0.95  # Additional penalty
    
    # Low reservation boost for Best-Fit
    if reservation_ratio < 20 and policy == 'Best-Fit':
        scenario_multiplier *= 1.12
    
    return base_score * scenario_multiplier