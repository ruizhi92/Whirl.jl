module TimeMarching

import Whirl:@get

export System, Constrained, Unconstrained, IFRK, IFHERK

"Abstract type for a system of ODEs"
abstract type System{C} end

const Constrained = true
const Unconstrained = false

#function A⁻¹ end
#function r₁ end

struct RKParams{N}
  c::Vector{Float64}
  a::Matrix{Float64}
end

using ..SaddlePointSystems

const RK31 = RKParams{3}([0.5, 1.0, 1.0],
                      [1/2        0        0
                       √3/3 (3-√3)/3        0
                       (3+√3)/6    -√3/3 (3+√3)/6])

const Euler = RKParams{1}([1.0],ones(1,1))


include("timemarching/ifrk.jl")

include("timemarching/ifherk.jl")

include("timeMarching/herkbody.jl")

#=

struct Operators{TA}

  A⁻¹ :: TA
  r₁ :: Function

end

struct ConstrainedOperators{TA,S,S0}

  A⁻¹ :: TA
  B₁ᵀ :: Function
  B₂ :: Function
  P  :: Function  # Smoother of constraint data (or identity)
  S⁻¹ :: S
  S₀⁻¹ :: S0
  r₁ :: Function
  r₂ :: Function

end


struct TimeParams
  Δt::Float64
  rk::RKParams
end

function Base.show(io::IO, p::TimeParams)
    print(io, "Time step size $(p.Δt)")
end
=#
#=
  We seek to advance, from tⁿ to tⁿ+Δt, the solution of equations of the form

  d/dt (Hu) + HB₁ᵀf =  Hr₁(u,t)

  subject to the constraint

  B₂u = r₂(t)

  where H is the integrating factor H(tⁿ+Δt-t). Each stage of the IF-HERK algorithm
  requires the solution of the saddle point problem

   [Aⁱ  B₁ᵀ](u) = (R₁)
   [B₂ 0  ](f̃) = (R₂)

  where Aⁱ = H(t(i)-t(i+1)). Ultimately, this is solved by use of the Schur complement
  Sⁱ = -B₂(Aⁱ)⁻¹B₁ᵀ. Note that only (Aⁱ)⁻¹ is required, not Aⁱ itself.

  We assume that the algorithm here only has two different forms of (Aⁱ)⁻¹ in the
  various stages: H(Δt/2) and H(0) = I (identity). The former is referred to as
  A⁻¹ in the code. The solution at tⁿ is delivered in the structure `s`, and
  returned as an updated version of itself at tⁿ+Δt. The time parameters are
  provided in `p`. The inverses of the Schur complements corresponding to
  H(Δt/2) and to H(0) are provided as operators S⁻¹ and S₀⁻¹, respectively.

  A⁻¹ is function that acts upon data of size s.u and returns data of same size
  B₁ᵀ is function that acts upon data of size s.f and returns data of size s.u
  B₂ is a function that acts upon data of size s and returns data of size s.f
  S and S₀ are factorized Schur complement matrices
  r₁ is function that acts upon solution structure s and returns data of size s.u
  r₂ is function that acts upon time value and returns data of size s.f
=#
#=
function ifherk!(s::Whirl.ConstrainedSoln{T,K},p::TimeParams,ops::ConstrainedOperators) where {T,K}
# Advance the solution by one time step
@get p (Δt,rk)
@get ops (A⁻¹,B₁ᵀ,B₂,P,S⁻¹,S₀⁻¹,r₁,r₂)

# first stage
sᵢ = deepcopy(s)
sᵢ₊₁ = deepcopy(sᵢ)
sᵢ₊₁.t = s.t + Δt*rk.c[1]
A⁻¹gᵢ = Δt*rk.a[1][1]*A⁻¹(r₁(sᵢ,sᵢ₊₁.t))
qᵢ₊₁ = A⁻¹(s.u)
sᵢ₊₁.u = qᵢ₊₁ + A⁻¹gᵢ
sᵢ₊₁.f = -P(S⁻¹(B₂(sᵢ₊₁.u) - r₂(sᵢ,sᵢ₊₁.t)))
A⁻¹B₁ᵀf = A⁻¹(B₁ᵀ(sᵢ₊₁.f))
@. sᵢ₊₁.u -= A⁻¹B₁ᵀf

w = []
for i = 2:rk.nstage-1
  sᵢ = deepcopy(sᵢ₊₁)
  sᵢ₊₁.t = s.t + Δt*rk.c[i]
  push!(w,(A⁻¹gᵢ-A⁻¹B₁ᵀf)/(Δt*rk.a[i-1][i-1]))
  for j = 1:i-1
    w[j] = A⁻¹(w[j])
  end
  A⁻¹gᵢ = Δt*rk.a[i][i]*A⁻¹(r₁(sᵢ,sᵢ₊₁.t))
  qᵢ₊₁ = A⁻¹(qᵢ₊₁)
  @. sᵢ₊₁.u = qᵢ₊₁ + A⁻¹gᵢ
  for j = 1:i-1
    @. sᵢ₊₁.u += Δt*rk.a[i][j]*w[j]
  end
  sᵢ₊₁.f = -P(S⁻¹(B₂(sᵢ₊₁.u) - r₂(sᵢ,sᵢ₊₁.t)))
  A⁻¹B₁ᵀf = A⁻¹(B₁ᵀ(sᵢ₊₁.f))
  @. sᵢ₊₁.u -= A⁻¹B₁ᵀf
end

# In final stage, A⁻¹ is assumed to be the identity
i = rk.nstage
sᵢ = deepcopy(sᵢ₊₁)
sᵢ₊₁.t = s.t + Δt*rk.c[i]
push!(w,(A⁻¹gᵢ-A⁻¹B₁ᵀf)/(Δt*rk.a[i-1][i-1]))
for j = 1:i-1
  w[j] = w[j]
end
A⁻¹gᵢ = Δt*rk.a[i][i]*r₁(sᵢ,sᵢ₊₁.t)
sᵢ₊₁.u = qᵢ₊₁ + A⁻¹gᵢ
for j = 1:i-1
  @. sᵢ₊₁.u += Δt*rk.a[i][j]*w[j]
end
sᵢ₊₁.f = -P(S₀⁻¹(B₂(sᵢ₊₁.u) - r₂(sᵢ,sᵢ₊₁.t)))
A⁻¹B₁ᵀf = B₁ᵀ(sᵢ₊₁.f)
@. sᵢ₊₁.u -= A⁻¹B₁ᵀf

# Finalize
s = deepcopy(sᵢ₊₁)
s.f /= Δt*rk.a[rk.nstage][rk.nstage]

return s

end

function ifrk!(s₊, s, Δt, rk::RKParams, sys::System{Unconstrained})
    @get sys (A⁻¹g, q, Ñ, w) # scratch space
    resize!(w, rk.nstage-1)

    u₊ = s₊.u
    u₊ .= s.u

    t₊ = s.t + Δt*rk.c[1]

    A⁻¹(A⁻¹g, r₁(Ñ, u₊, t₊, sys), sys)
    A⁻¹g .*= Δt*rk.a[1, 1]

    A⁻¹(q, s.u, sys)

    @. u₊ = q + A⁻¹g

    for i = 2:rk.nstage-1
        t₊ = s.t + Δt*rk.c[i]

        w[i-1] .= A⁻¹g ./ (Δt*rk.a[i-1, i-1])
        for j = 1:i-1
            A⁻¹(w[j], w[j], sys)
        end

        A⁻¹(A⁻¹g, r₁(Ñ, u₊, t₊, sys), sys)
        A⁻¹g .*= Δt*rk.a[i,i]

        A⁻¹(q, q, sys)

        @. u₊ = q + A⁻¹g

        for j = 1:i-1
            @. u₊ += Δt*rk.a[i,j]*w[j]
        end
    end

    # In final stage, A⁻¹ is assumed to be the identity
    i = rk.nstage
    t₊ = s.t + Δt*rk.c[i]
    w[i-1] .= A⁻¹g ./ (Δt*rk.a[i-1,i-1])

    r₁(A⁻¹g, u₊, t₊, sys)
    A⁻¹g .*= Δt*rk.a[i,i]

    @. u₊ = q + A⁻¹g
    for j in 1:i-1
        u₊ .+= Δt*rk.a[i,j].*w[j]
    end

    s₊.t = t₊

    return s₊
end

=#

end
