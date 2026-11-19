## Tone-shift check on study images

images: 360 (msid: 180, msld_v2: 180); shifts attempted: 1080
skipped: 1.7% (near-neutral colour: 18)
healthy-skin ITA after shift vs target: median absolute error 1.1 deg; lands in the target band 92.2% of the time
wanted change hit the 35-unit cap: 5.4%
median dL* healthy skin: 13.1; median |dL*| lesion pixels: 1.9

### Band hit rate (%), original band (rows) x target band (columns)
target_band        dark intermediate      light
orig_band                                      
dark              100.0    85.714286  68.907563
intermediate  98.333333    98.333333  94.166667
light         88.695652    95.652174      100.0

### Skipped (%) by original band
orig_band
dark            0.8
intermediate    0.0
light           4.2

### Share of pixels detected as skin (median) by original band
orig_band
dark            0.808
intermediate    0.865
light           0.824

### Median absolute ITA error (deg) by target band
target_band
dark            1.3
intermediate    1.1
light           1.0
