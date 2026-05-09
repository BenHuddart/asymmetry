import warnings
import numpy as np
from asymmetry.core.fitting.models import longitudinal_field_kubo_toyabe
import time

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # Test with challenging parameters
    t = np.linspace(0, 20, 200)
    
    # First call (will compute integrals)
    start = time.time()
    result1 = longitudinal_field_kubo_toyabe(t, A0=1.0, Delta=0.3, B_L=0.05, baseline=0.0)
    time1 = time.time() - start
    
    # Second call with same parameters (should use cache)
    start = time.time()
    result2 = longitudinal_field_kubo_toyabe(t, A0=1.0, Delta=0.3, B_L=0.05, baseline=0.0)
    time2 = time.time() - start
    
    # Check results
    print(f"First call: {time1:.4f}s, Second call (cached): {time2:.4f}s")
    print(f"Results equal: {np.allclose(result1, result2)}")
    print(f"Results finite: {np.all(np.isfinite(result1))}")
    
    # Check warnings
    integration_warnings = [warning for warning in w if 'IntegrationWarning' in str(warning.category)]
    print(f"Integration warnings: {len(integration_warnings)}")
    for warning in integration_warnings[:3]:
        print(f"  - {warning.message}")
