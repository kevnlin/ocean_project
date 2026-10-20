# Training window × model size, CESM2, Argo + surface inputs

Zihao's shared T/S alignment model (`profile_shared`, Local Transformer, 6 blocks, 8 heads) with Argo plus SST, SSS and steric SSH, the local OI-error scale by depth × 5° cell, 15,000 steps of 1,024 queries, checkpoint chosen on 2004, scored on 2005. Training windows end in 2003; the normalisation, climatology, OI anchor and scale table are the 2000–2003 fits for every window, so only the number of training months changes (48, 120, 240 or 480). Steric SSH is computed from the same T/S that is reconstructed and SST/SSS are its top level, without noise: an optimistic stand-in for satellite data.

## 2005 test RMSE

Mean ± standard deviation over seeds; temperature °C / salinity PSU. Frozen OI on the same cells: 0.0860 °C / 0.0226 PSU.

| Training window | s: 64 slots, width 192 | m: 128 slots, width 256 | l: 256 slots, width 384 |
|---|---|---|---|
| 4 years | 0.0654 ± 0.0009 / 0.0162 ± 0.0004 | 0.0643 ± 0.0003 / 0.0159 ± 0.0002 | 0.0664 ± 0.0016 / 0.0168 ± 0.0004 |
| 10 years | 0.0662 ± 0.0004 / 0.0166 ± 0.0003 | 0.0652 ± 0.0001 / 0.0163 ± 0.0001 | 0.0662 ± 0.0008 / 0.0168 ± 0.0002 |
| 20 years | 0.0657 ± 0.0002 / 0.0164 ± 0.0003 | 0.0649 ± 0.0003 / 0.0161 ± 0.0002 | 0.0659 ± 0.0013 / 0.0165 ± 0.0005 |
| 40 years | 0.0660 ± 0.0005 / 0.0168 ± 0.0003 | 0.0658 ± 0.0006 / 0.0166 ± 0.0004 | 0.0669 ± 0.0003 / 0.0168 ± 0.0002 |

## Change of the 2005 RMSE

Against the 4-year window of the same size, then against OI; temperature / salinity.

| Training window | s | m | l |
|---|---|---|---|
| 4 years | –; vs OI -23.9 % / -28.4 % | –; vs OI -25.2 % / -29.9 % | –; vs OI -22.7 % / -25.7 % |
| 10 years | +1.2 % / +2.1 %; vs OI -23.0 % / -26.8 % | +1.4 % / +2.7 %; vs OI -24.1 % / -28.0 % | -0.3 % / -0.3 %; vs OI -23.0 % / -25.9 % |
| 20 years | +0.4 % / +0.9 %; vs OI -23.6 % / -27.7 % | +0.9 % / +1.3 %; vs OI -24.5 % / -29.0 % | -0.8 % / -2.0 %; vs OI -23.3 % / -27.2 % |
| 40 years | +0.9 % / +3.7 %; vs OI -23.2 % / -25.7 % | +2.3 % / +4.7 %; vs OI -23.5 % / -26.6 % | +0.8 % / -0.1 %; vs OI -22.2 % / -25.7 % |

## Training speed

Steps to reach the 2004 score the 4-year, smallest model ends with (0.2615, mean per-variable standardized RMSE), per seed; – if not reached in 15,000 steps.

| Training window | s | m | l |
|---|---|---|---|
| 4 years | 14.7k, 14.8k, – | 12.0k, 12.4k, 12.6k | –, –, – |
| 10 years | –, –, 14.5k | 13.2k, 13.9k, 12.5k | –, –, – |
| 20 years | –, –, – | –, 14.8k, 14.9k | –, –, 14.2k |
| 40 years | –, –, – | –, –, – | –, –, – |

2004 score (mean per-variable standardized RMSE) after a number of steps, mean over seeds:

| Window, size | 1k | 3k | 5k | 10k | 15k | best step |
|---|---|---|---|---|---|---|
| 4 y, s | 0.3034 | 0.2873 | 0.2790 | 0.2667 | 0.2615 | 15000, 15000, 15000 |
| 4 y, m | 0.3047 | 0.2876 | 0.2777 | 0.2666 | 0.2590 | 15000, 15000, 15000 |
| 4 y, l | 0.3070 | 0.2912 | 0.2831 | 0.2710 | 0.2656 | 15000, 15000, 14000 |
| 10 y, s | 0.3078 | 0.2864 | 0.2783 | 0.2664 | 0.2628 | 15000, 14000, 15000 |
| 10 y, m | 0.3061 | 0.2869 | 0.2780 | 0.2639 | 0.2601 | 15000, 14000, 15000 |
| 10 y, l | 0.3097 | 0.2933 | 0.2818 | 0.2670 | 0.2640 | 15000, 14000, 15000 |
| 20 y, s | 0.3085 | 0.2899 | 0.2811 | 0.2701 | 0.2646 | 15000, 15000, 15000 |
| 20 y, m | 0.3062 | 0.2910 | 0.2822 | 0.2691 | 0.2628 | 15000, 15000, 15000 |
| 20 y, l | 0.3100 | 0.2943 | 0.2856 | 0.2694 | 0.2635 | 13000, 14000, 15000 |
| 40 y, s | 0.3079 | 0.2933 | 0.2827 | 0.2724 | 0.2678 | 15000, 14000, 14000 |
| 40 y, m | 0.3142 | 0.2934 | 0.2848 | 0.2731 | 0.2663 | 14000, 14000, 13000 |
| 40 y, l | 0.3134 | 0.2978 | 0.2890 | 0.2742 | 0.2669 | 15000, 14000, 13000 |

## Cost

| Window, size | parameters | hours a run | peak GPU memory |
|---|---|---|---|
| 4 y, s | 5.21 M | 1.8 ± 0.0 | 8.8 GB |
| 4 y, m | 9.23 M | 1.9 ± 0.0 | 9.6 GB |
| 4 y, l | 20.67 M | 2.0 ± 0.0 | 11.2 GB |
| 10 y, s | 5.21 M | 1.8 ± 0.0 | 8.8 GB |
| 10 y, m | 9.23 M | 1.8 ± 0.0 | 9.6 GB |
| 10 y, l | 20.67 M | 2.0 ± 0.0 | 11.2 GB |
| 20 y, s | 5.21 M | 1.9 ± 0.1 | 8.8 GB |
| 20 y, m | 9.23 M | 2.0 ± 0.0 | 9.6 GB |
| 20 y, l | 20.67 M | 2.0 ± 0.0 | 11.2 GB |
| 40 y, s | 5.21 M | 1.9 ± 0.0 | 8.8 GB |
| 40 y, m | 9.23 M | 2.0 ± 0.0 | 9.6 GB |
| 40 y, l | 20.67 M | 2.0 ± 0.0 | 11.2 GB |

## 30,000 steps

The 4- and 40-year windows trained twice as long (the learning-rate schedule spans 30,000 steps); everything else as above. 2005 RMSE, temperature °C / salinity PSU, mean ± sd over seeds; then the change against the 15,000-step run of the same cell.

| Training window | s: 64 slots, width 192 | m: 128 slots, width 256 | l: 256 slots, width 384 |
|---|---|---|---|
| 4 years | 0.0612 ± 0.0011 / 0.0153 ± 0.0004 | 0.0604 ± 0.0003 / 0.0155 ± 0.0004 | 0.0613 ± 0.0009 / 0.0154 ± 0.0003 |
| 40 years | 0.0612 ± 0.0011 / 0.0153 ± 0.0003 | 0.0615 ± 0.0011 / 0.0156 ± 0.0001 | 0.0635 ± 0.0004 / 0.0158 ± 0.0003 |

| Change against 15,000 steps | s | m | l |
|---|---|---|---|
| 4 years | -6.4 % / -5.7 % | -6.0 % / -2.6 % | -7.7 % / -8.3 % |
| 40 years | -7.3 % / -8.9 % | -6.6 % / -6.1 % | -5.2 % / -5.9 % |

40 years against 4 years at 30,000 steps: s +0.0 % / +0.2 %; m +1.7 % / +0.9 %; l +3.5 % / +2.5 %

Steps to the 2004 target of the 15,000-step section (0.2615), per seed; best step:

| Window, size | steps to target | 5k | 10k | 15k | 20k | 25k | 30k | best step |
|---|---|---|---|---|---|---|---|---|
| 4 y, s | 12.5k, 14.2k, 11.1k | 0.2784 | 0.2664 | 0.2565 | 0.2514 | 0.2481 | 0.2467 | 30000, 30000, 29000 |
| 4 y, m | 13.5k, 15.2k, 14.1k | 0.2786 | 0.2680 | 0.2600 | 0.2531 | 0.2494 | 0.2481 | 30000, 30000, 27000 |
| 4 y, l | 14.9k, 17.7k, 17.2k | 0.2842 | 0.2714 | 0.2628 | 0.2566 | 0.2515 | 0.2496 | 28000, 30000, 30000 |
| 40 y, s | 16.7k, 21.5k, 19.1k | 0.2843 | 0.2748 | 0.2664 | 0.2607 | 0.2551 | 0.2554 | 27000, 27000, 28000 |
| 40 y, m | 18.4k, 17.4k, 24.2k | 0.2868 | 0.2755 | 0.2666 | 0.2625 | 0.2561 | 0.2563 | 30000, 29000, 25000 |
| 40 y, l | 21.0k, 24.9k, 20.3k | 0.2943 | 0.2778 | 0.2694 | 0.2649 | 0.2580 | 0.2572 | 30000, 27000, 25000 |
