"""Print M#-aware capacity estimates for Fe:LiNbO3.

Uses the same geometry as the rest of the project and conservative
material parameters typical of iron-doped lithium niobate.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from atom.capacity import CapacityParams, capacity_summary, geometric_capacity, usable_capacity


def main() -> None:
    print("Fe:LiNbO3 capacity model (conservative 90-degree defaults)\n")

    base = CapacityParams()
    s = capacity_summary(base)

    print(f"Volume                : {s['volume_cm3']:.1f} cm3")
    print(f"M#                    : {s['m_number']}")
    print(f"eta_min               : {s['eta_min']:.0e}")
    print(f"Geometric channels    : {s['geometric_channels']:.0f}")
    print(f"Max usable channels   : {s['max_usable_channels']:.0f}")
    print()
    print(f"Geometric capacity    : {s['geometric_capacity']:.3e}")
    print(f"Usable capacity       : {s['usable_capacity']:.3e}")
    print(f"Usable / geometric    : {s['usable_fraction']:.3f}")
    print()

    print("Sensitivity to M# (1 cm3, eta_min=1e-4):")
    for m in (1.0, 2.0, 5.0, 10.0):
        p = CapacityParams(m_number=m)
        print(f"  M#={m:4.1f}  ->  {usable_capacity(p):.3e} values")

    print()
    print("Volume scaling at default M#=2.0:")
    for side in (1.0, 2.0, 3.0):
        p = CapacityParams(side_cm=side)
        print(f"  {side:.0f} cm side ({p.volume_cm3():.0f} cm3)  ->  {usable_capacity(p):.3e} values")


if __name__ == "__main__":
    main()
