# BPP relaxation: comparison

| Aspect | musrfit | WiMDA | Mantid | Asymmetry |
|---|---|---|---|---|
| Dedicated BPP parametric model | ❌ | ❌ | ❌ | ❌ |
| Workaround | FUNCTIONS block | external | external | scipy |

## Form

The model:

.. math::

   \lambda(T) = \lambda_0\, \frac{\tau(T)}{1 + (\omega \tau(T))^2},
   \quad \tau(T) = \tau_0\, e^{E_a / k_B T}.

Parameters: λ₀ (high-T limit), τ₀ (attempt time), E_a (activation
energy), ω (Larmor angular frequency).

Implementation: ~30 lines of numpy in `core/fitting/diffusion.py`
or as a new module. Drop into the parameter-trending registry.
