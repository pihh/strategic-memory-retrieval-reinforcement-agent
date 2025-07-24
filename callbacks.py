class EarlyStoppingMonitor:
    """
    Utility to monitor performance (e.g., reward) and signal early stopping 
    when a target threshold is consistently reached.

    Features:
        - Keeps a moving window of recent scores/rewards.
        - Triggers stopping if all scores in the window exceed a specified target.
        - Can be used to terminate training once performance is "good enough".

    Args:
        target (float): The reward/performance threshold to achieve.
        min_logs (int): Minimum number of recent results to consider for stopping.

    Example:
        monitor = EarlyStoppingMonitor(target=0.95, min_logs=5)
        for episode in range(1000):
            ...
            done = monitor.update(mean_reward)
            if done:
                print("Early stopping triggered!")
                break
    """
    def __init__(self, target=0.95, min_logs=5):
        self.target = target        # Reward threshold to trigger stopping
        self.min_logs = min_logs    # Number of recent results required
        self.history = []           # Stores recent rewards/scores

    def update(self, rew):
        """
        Add a new reward/score and check if early stopping criteria is met.
        
        Args:
            rew (float): The latest reward or performance metric.

        Returns:
            bool: True if early stopping should be triggered, else False.
        """
        self.history.append(rew)
        if len(self.history) > self.min_logs:
            self.history.pop(0)  # Maintain window size
        return self.should_stop()

    def should_stop(self):
        """
        Check if early stopping criteria are satisfied.

        Returns:
            bool: True if all recent rewards >= target and enough data has been logged.
        """
        if len(self.history) < self.min_logs:
            return False
        # All recent rewards must meet or exceed the target
        return all(r >= self.target for r in self.history)


def print_sb3_style_log_box(stats):
    """
    Prints a multi-section summary table in the style of SB3 (Stable-Baselines3) logs.
    Output is boxed, sectioned, and auto-aligned, with centered numbers and closing bars.
    
    Args:
        stats (list of dict): 
            Each dict represents a section, with:
                - "header": str, section header name
                - "stats": dict of (label: value) pairs

    Example:
        stats = [
            {
                "header": "rollout",
                "stats": dict(ep_len_mean=8.0, ep_rew_mean=0.98)
            },
            ...
        ]
        print_sb3_style_log_box(stats)
    """
    # Flatten all rows to compute correct widths for box alignment
    all_rows = []
    for section in stats:
        all_rows.append((section["header"] + "/", None, True))  # Section header
        for k, v in section["stats"].items():
            all_rows.append((k, v, False))                      # Stat row

    # Compute the key (label) width for best alignment
    key_width = max(
        len("    " + k) if not is_section else len(k)
        for k, v, is_section in all_rows
    )
    val_width = 10  # Standard SB3 width for numbers (centered in column)
    box_width = 2 + key_width + 1 + val_width + 2  # | key | val |

    def fmt_row(label, value, is_section):
        """
        Formats a single row (header or value) with correct alignment and padding.
        """
        if is_section:
            # Section header: left-aligned label, blank value field, both with bars
            return f"| {label:<{key_width}}|{' ' * val_width} |"
        else:
            # Stat line: label left-aligned (with indent), value centered
            # Support PyTorch .item() for tensor values
            if hasattr(value, 'item'):
                value = value.item()
            # Format number
            if isinstance(value, float):
                s_value = f"{value:8.3f}"
            elif isinstance(value, int):
                s_value = f"{value:8d}"
            else:
                s_value = str(value)
            # Center value in value column (for classic SB3 look)
            s_value_centered = f"{s_value:^{val_width}}"
            return f"|    {label:<{key_width-4}} |{s_value_centered} |"

    # Print top border
    print("-" * box_width)
    # Print each section
    for section in stats:
        print(fmt_row(section["header"] + "/", None, True))
        for k, v in section["stats"].items():
            print(fmt_row(k, v, False))
    # Print bottom border
    print("-" * box_width)
