include(Pkg.dir("Dyn3d")*"/src/Dyn3d.jl")
using Dyn3d

"""
    A(body_dyn)

Returns the collected inertia matrix of all body in their own body coord.

# Arguments

    - 'body_dyn' : a stuct containing all state information of bodys and joints

"""
function A(body_dyn::BodyDyn)
    @get body_dyn (sys, )

    return sys.Ib_total
end

"""
    B₁ᵀ(body_dyn)

Computes the constraint matrix for Lagrange multiplier force term. B₁ᵀ reveals
the active-reactive forcing term on parent-child body.

# Arguments

    - 'body_dyn' : a stuct containing all state information of bodys and joints

"""
function B₁ᵀ(body_dyn::BodyDyn)
    @get body_dyn (bs, sys)
    # pointer to pre-allocated array
    @get sys.pre_array (A_total,)

    # construct A_total to take in parent-child hierarchy
    for i = 1:sys.nbody
        # fill in parent joint blocks
        A_total[6i-5:6i, 6i-5:6i] = eye(Float64, 6)
        # fill in child joint blocks except for those body whose nchild=0
        for child_count = 1:bs[i].nchild
            chid = bs[i].chid[child_count]
            A_total[6i-5:6i, 6chid-5:6chid] = - (bs[chid].Xp_to_b)'
        end
    end
    return A_total*sys.T_total
end

"""
    B₂(body_dyn)

Computes the motion constraint matrix acting on all body's velocity. These
constraints arise from body velocity relation in each body's local body coord,
for example if body 2 and 3 are connected then:
    v(3) = vJ(3) + X2_to_3*v(2)
"""
function B₂(body_dyn::BodyDyn)
    @get body_dyn (bs, sys)
    # pointer to pre-allocated array
    @get sys.pre_array (B_total,)

    # construct B_total to take in parent-child hierarchy
    for i = 1:sys.nbody
        # fill in child body blocks
        B_total[6i-5:6i, 6i-5:6i] = eye(Float64, 6)
        # fill in parent body blocks except for those body whose pid=0
        if bs[i].pid != 0
            pid = bs[i].pid
            B_total[6i-5:6i, 6pid-5:6pid] = - bs[i].Xp_to_b
        end
    end
    return (sys.T_total')*B_total
end

"""
    r₁(body_dyn)

Computes the right hand side of body dynamics momentum equation using body
velocity and joints position. It returns a forcing term, which is a summation of
bias force term and joint spring-damper forcing term. The bias term includes the
change of inertia effect, together with gravity and external force.

# Arguments

- 't' : time to evalute r₁
- 'body_dyn' : a stuct containing all state information of bodys and joints

"""
function r₁(body_dyn::BodyDyn)
    @get body_dyn (bs, js, sys)
    # pointer to pre-allocated array
    @get sys.pre_array (p_total, τ_total, p_bias, f_g, f_ex, r_temp,
        Xic_to_i, A_total)

    # compute bias force, gravity and external force
    for i = 1:sys.nbody
        # bias force
        p_bias = Mfcross(bs[i].v, (bs[i].inertia_b*bs[i].v))
        # gravity in inertial center coord
        f_g = bs[i].mass*[zeros(T, 3); sys.g]
        # get transform matrix from x_c in inertial frame to the origin of
        # inertial frame
        r_temp = [zeros(T, 3); -bs[i].x_c]
        r_temp = bs[i].Xb_to_i*r_temp
        r_temp = [zeros(T, 3); -bs[i].x_i + r_temp[4:6]]
        Xic_to_i = TransMatrix(r_temp)
        # transform gravity force
        f_g = bs[i].Xb_to_i'*inv(Xic_to_i')*f_g
        # external force described in inertial coord
        f_ex = zeros(T, 6)
        f_ex = bs[i].Xb_to_i*f_ex
        # add up
        p_total[6i-5:6i] = p_bias - (f_g + f_ex)
    end

    # construct τ_total, this is related only to spring force.
    # τ is only determined by whether the dof has resistance(damp and
    # stiff) or not. Both active dof and passive dof can have τ term
    for i = 1:sys.nbody, k = 1:js[i].nudof
        # find index of the dof in the unconstrained list of this joint
        dofid = js[i].joint_dof[k].dof_id
        τ_total[js[i].udofmap[k]] = -js[i].joint_dof[k].stiff*js[i].qJ[dofid] -
                                    js[i].joint_dof[k].damp*js[i].vJ[dofid]
    end

    # construct A_total to take in parent-child hierarchy
    for i = 1:sys.nbody
        # fill in parent joint blocks
        A_total[6i-5:6i, 6i-5:6i] = eye(T, 6)
        # fill in child joint blocks except for those body whose nchild=0
        for child_count = 1:bs[i].nchild
            chid = bs[i].chid[child_count]
            A_total[6i-5:6i, 6chid-5:6chid] = - bs[chid].Xp_to_b
        end
    end

    # collect all together
    return -p_total + A_total*sys.S_total*τ_total
end

"""
    r₂(t, body_dyn)

Computes the right hand side of joint velocity constraint equation. Return
relative velocity relation of constrained dofs between connected joints,
based on their parent-child hierarchy. If a joint has active motion, then a
non-zero value is given. Otherwise the value is 0 for this particular dof.

# Arguments

- 't' : time to evalute r₂
- 'body_dyn' : a stuct containing all state information of bodys and joints

"""
function r₂(t::T, body_dyn::BodyDyn) where T <: AbstractFloat
    @get body_dyn (js, sys)
    # pointer to pre-allocated array
    @get sys.pre_array (v_gti, va_gti)

    # give actual numbers from calling motion(t)
    for i = 1:sys.na
        jid = sys.kinmap[i,1]
        dofid = sys.kinmap[i,2]
        _, va_gti[i] = js[jid].joint_dof[dofid].motion(t)
    end

    v_gti[sys.udof_a] = va_gti
    return (sys.T_total')*v_gti
end
