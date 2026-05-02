import numpy as np
from collections import Counter

def generate_arrivals(lambda_func, start=0, end=600):
    """
    Generate arrivals using thinning algorithm for non-homogeneous Poisson process.
    """
    arrivals = []
    t = start
    time_points = np.linspace(start, end, 1000)
    lambda_max = max(lambda_func(time_points))
    
    while t < end:
        t += np.random.exponential(1 / lambda_max)
        if t > end:
            break
        if np.random.random() < lambda_func(t) / lambda_max:
            arrivals.append(t)
    return np.array(arrivals)


def service_time():
    """Lognormal distribution with mean ≈ 70 minutes."""
    sigma = 0.35
    mu = np.log(70) - 0.5 * sigma**2
    sample = np.random.lognormal(mu, sigma)
    return sample


def party_size():
    """Generate random party size based on realistic distribution."""
    sizes = [1, 2, 3, 4, 5, 6]
    probs = [0.1, 0.45, 0.15, 0.2, 0.06, 0.04]
    return np.random.choice(sizes, p=probs)


def _cap_for_party(p):
    """Find smallest table capacity that fits party."""
    if p <= 2:
        return 2
    elif p <= 4:
        return 4
    else:
        return 6


def seat_party(policy, free_tables, party, current_time, reservations,
               tables_total=None, holdback_window=60):
    """
    Decide which table to assign based on the policy.
    """
    # ---------- Best-Fit ----------
    if policy == "Best-Fit":
        for cap in sorted(free_tables.keys()):
            if cap >= party and free_tables[cap] > 0:
                return cap
        return None

    # ---------- Hold-Back ----------
    elif policy == "Hold-Back":
        upcoming = [r for r in reservations 
                    if 0 <= r[0] - current_time <= holdback_window]
        
        if not upcoming:
            for cap in sorted(free_tables.keys()):
                if cap >= party and free_tables[cap] > 0:
                    return cap
            return None
        
        need_by_cap = Counter(_cap_for_party(r[1]) for r in upcoming)
        
        protected = {}
        for cap in free_tables:
            total_tables = tables_total.get(cap, free_tables[cap]) if tables_total else free_tables[cap]
            reservations_needed = need_by_cap.get(cap, 0)
            
            max_holdback = min(
                reservations_needed,
                max(0, free_tables[cap] // 2),
                max(0, total_tables - 1)
            )
            protected[cap] = max_holdback
        
        for cap in sorted(free_tables.keys()):
            available_beyond_holdback = free_tables[cap] - protected.get(cap, 0)
            if cap >= party and available_beyond_holdback > 0:
                return cap
        
        earliest_reservation = min(r[0] for r in upcoming) if upcoming else float('inf')
        time_until_first_reservation = earliest_reservation - current_time
        
        if time_until_first_reservation > 30:
            for cap in sorted(free_tables.keys()):
                if cap >= party and free_tables[cap] > 0:
                    return cap
        
        return None

    # ---------- Overbooking ----------
    elif policy == "Overbooking":
        for cap in sorted(free_tables.keys()):
            if cap >= party and free_tables[cap] > 0:
                return cap
        
        if tables_total:
            total_capacity = sum(tables_total.values())
            occupied = total_capacity - sum(free_tables.values())
            occupancy_rate = occupied / total_capacity if total_capacity > 0 else 0
            
            base_prob = 0.15
            
            if occupancy_rate > 0.8:
                base_prob = 0.30
            elif occupancy_rate > 0.6:
                base_prob = 0.20
            
            if party <= 2:
                base_prob *= 1.5
            elif party >= 5:
                base_prob *= 0.7
            
            acceptance_prob = min(base_prob, 0.40)
            
            if np.random.random() < acceptance_prob:
                return "virtual"
        else:
            if np.random.random() < 0.15:
                return "virtual"
        
        return None

    return None
