# utils/schedules.py
import math

def linear_ramp(epoch, warmup_epochs, target):
    if warmup_epochs<=0: return target
    return min(target, target * (epoch+1)/warmup_epochs)

def cosine_ramp(epoch, start_epoch, end_epoch, max_val):
    if epoch < start_epoch: return 0.0
    t = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    t = min(max(t, 0.0), 1.0)
    return max_val * 0.5 * (1 - math.cos(math.pi * t))
