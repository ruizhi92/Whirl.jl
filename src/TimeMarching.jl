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
include(Pkg.dir("Dyn3d")*"/src/Dyn3d.jl")
using Dyn3d

const RK31 = RKParams{3}([0.5, 1.0, 1.0],
                      [1/2        0        0
                       √3/3 (3-√3)/3        0
                       (3+√3)/6    -√3/3 (3+√3)/6])

const Euler = RKParams{1}([1.0],ones(1,1))


include("timemarching/ifrk.jl")

include("timemarching/ifherk.jl")

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

mutable struct HERKBodyOnly{NS,FA,FB1,FB2,FR1,FR2,TU,TF}

  # time step size
  Δt :: Float64

  rk :: RKParams

  A :: FA
  B₁ᵀ :: FB1  # operates on TF and returns TU
  B₂ :: FB2   # operates on TU and returns TF
  r₁ :: FR1  # function of u and t, returns TU
  r₂ :: FR2  # function of t, returns TF

  # Saddle-point systems
  S :: Vector{SaddleSystem}  # -B₂HB₁ᵀ

  # scratch space
  # qᵢ :: TU
  ubuffer :: TU   # should not need this
  # w :: Vector{TU}
  fbuffer :: TF

  # flags
  _issymmetric :: Bool

end

function (::Type{HERKBodyOnly})(u::TU,f::TF,Δt::Float64,
                          A::FA,B₁ᵀ::FB1,B₂::FB2,r₁::FR1,r₂::FR2;
                          issymmetric::Bool=false,
                          rk::RKParams{NS}=RK31) where {TU,TF,FA,FB1,FB2,FR1,FR2,NS}

    # scratch space
    qᵢ = deepcopy(u)
    ubuffer = deepcopy(u)
    w = [deepcopy(u) for i = 1:NS-1]
    fbuffer = deepcopy(f)

    dclist = diff([0;rk.c])

    # construct an array of operators for the integrating factor. Each
    # one can act on data of type `u` and return data of the same type.
    # e.g. we can call Hlist[1](u) to get the result.
    Hlist = [u -> plan_intfact(dc*Δt,u)*u for dc in unique(dclist)]

    Slist = [SaddleSystem(u,f,H,B₁ᵀ,B₂,issymmetric=issymmetric,isposdef=true) for H in Hlist]

    H = [Hlist[i] for i in indexin(dclist,unique(dclist))]
    S = [Slist[i] for i in indexin(dclist,unique(dclist))]

    htype,_ = typeof(H).parameters

    HERKBodyOnly{NS,FA,FB1,FB2,FR1,FR2,TU,TF}(Δt,rk,
                                H,B₁ᵀ,B₂,r₁,r₂,S,
                                qᵢ,ubuffer,w,fbuffer,
                                issymmetric)
end

function Base.show(io::IO, scheme::IFHERK{NS,FH,FB1,FB2,FR1,FR2,TU,TF}) where {NS,FH,FB1,FB2,FR1,FR2,TU,TF}
    println(io, "Order-$NS IF-HERK system with")
    println(io, "   State of type $TU")
    println(io, "   Force of type $TF")
    println(io, "   Time step size $(scheme.Δt)")
end


function HERK!(sᵢₙ::Soln{T}, bs::Vector{SingleBody}, js::Vector{SingleJoint},
    sys::System) where T <: AbstractFloat
"""
    HERKMain is a half-explicit Runge-Kutta solver based on the
    constrained body model in paper of V.Brasey and E.Hairer.
    The following ODE system is being solved:
       | dq/dt = v                     |
       | M(q)*dv/dt = f(q,v) - GT(q)*λ |
       | 0 = G(q)*v + gti(q)           |
    , where GT is similar to the transpose of G.

    Note that this is a index-2 system with two variables q and u.
    Here we denote for a body chain, v stands for body velocity
    and vJ stands for joint velocity. Similarly for q and qJ. λ is the
    constraint on q to be satisfied.

    Specificlly for the dynamics problem, we choose to solve for qJ and v.
    The system of equation is actually:
       | dqJ/dt = vJ                              |
       | M(qJ)*dv/dt = f(qJ,v,vJ) - GT(qJ)*lambda |
       | 0 = G(qJ)*v + gti(qJ)                    |
    So we need a step to calculate v from vJ solved. The motion constraint
    (prescribed active motion) is according to joint, not body.
"""
    @get sys.num_params (scheme, tol)

    # pick sheme parameters
    A, b, c, st = HERKScheme(scheme)
    if st != sys.num_params.st error("Scheme stage not correctly specified") end

    qJ_dim = sys.ndof
    λ_dim =sys.ncdof_HERK

    # pointer to pre-allocated array
    @get sys.pre_array (qJ, vJ, v, v̇, λ, v_temp, Mᵢ₋₁, fᵢ₋₁, GTᵢ₋₁, Gᵢ, gtiᵢ,
        lhs, rhs)

    # stage 1
    tᵢ₋₁ = sᵢₙ.t; tᵢ = sᵢₙ.t;; dt = sᵢₙ.dt
    qJ[1,:] = sᵢₙ.qJ
    v[1,:] = sᵢₙ.v
    # update vJ using v
    bs, js, sys, vJ[1,:] = UpdateVelocity!(bs, js, sys, v[1,:])

    # stage 2 to st+1
    for i = 2:st+1
        # time of i-1 and i
        tᵢ₋₁ = tᵢ
        tᵢ = sᵢₙ.t + dt*c[i]
        # initialize qJ[i,:]
        qJ[i,:] = sᵢₙ.qJ
        # calculate M, f and GT at tᵢ₋₁
        Mᵢ₋₁ = HERKFuncM(sys)
        fᵢ₋₁ = HERKFuncf(bs, js, sys)
        GTᵢ₋₁ = HERKFuncGT(bs, sys)
        # advance qJ[i,:]
        for k = 1:i-1
            qJ[i,:] += dt*A[i,k]*view(vJ,k,:)
        end
        # use new qJ to update system position
        bs, js, sys = UpdatePosition!(bs, js, sys, qJ[i,:])
        # calculate G and gti at tᵢ
        Gᵢ = HERKFuncG(bs, sys)
        gtiᵢ = HERKFuncgti(js, sys, tᵢ)
        # construct lhs matrix
        lhs = [ Mᵢ₋₁ GTᵢ₋₁; Gᵢ zeros(T,λ_dim,λ_dim) ]
        # the accumulated v term on the right hand side
        v_temp = sᵢₙ.v
        for k = 1:i-2
            v_temp += dt*A[i,k]*view(v̇,k,:)
        end
        # construct rhs
        rhs = [ fᵢ₋₁; -1./(dt*A[i,i-1])*(Gᵢ*v_temp + gtiᵢ) ]
######### use Julia's built in "\" operator for now
        # solve the eq
        x = lhs \ rhs
        # x = BlockLU(lhs, rhs, qJ_dim, λ_dim)
        # apply the solution
        v̇[i-1,:] = x[1:qJ_dim]
        λ[i-1,:] = x[qJ_dim+1:end]
        # advance v[i,:]
        v[i,:] = sᵢₙ.v
        for k = 1:i-1
            v[i,:] += dt*A[i,k]*view(v̇,k,:)
        end
        # update vJ using updated v
        bs, js, sys, vJ[i,:] = UpdateVelocity!(bs, js, sys, v[i,:])
# println("v = ", v[i,:])
    end

    # use norm(v[st+1,:]-v[st,:]) to determine next timestep
    sₒᵤₜ = Soln(tᵢ) # init struct
    sₒᵤₜ.dt = sᵢₙ.dt*(tol/norm(view(v,st+1,:)-view(v,st,:)))^(1/3)
    sₒᵤₜ.t = sᵢₙ.t + sᵢₙ.dt
    sₒᵤₜ.qJ = view(qJ, st+1, :)
    sₒᵤₜ.v = view(v, st+1, :)
    sₒᵤₜ.v̇ = view(v̇, st, :)
    sₒᵤₜ.λ = view(λ, st, :)

    return  sₒᵤₜ, bs, js, sys
end


end
