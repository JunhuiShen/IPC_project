#pragma once

#include "barrier_energy.h"
#include "contact_scheduling.h"

namespace solver_detail {

// Storage records contain only the fields requested by the solve. Constructors
// leave coefficients unwritten; the shared sparse evaluator assigns every field
// before publishing its activity byte, and discards contents on capacity growth.
struct RigidBlockContactTerms {
    Vec3 gradient;
    Mat33 hessian;
    RigidBlockContactTerms() {}
    template <RigidDerivativeMode Mode>
    void assign(const RigidEnergyDerivatives& value) {
        if constexpr (Mode == RigidDerivativeMode::TranslationHessian) {
            gradient = value.translation_gradient;
            hessian = value.translation_translation_hessian;
        } else {
            gradient = value.orientation_gradient;
            hessian = value.orientation_orientation_hessian;
        }
    }
    template <RigidDerivativeMode Mode>
    void add(RigidEnergyDerivatives& total) const {
        if constexpr (Mode == RigidDerivativeMode::TranslationHessian) {
            total.translation_gradient += gradient;
            total.translation_translation_hessian += hessian;
        } else {
            total.orientation_gradient += gradient;
            total.orientation_orientation_hessian += hessian;
        }
    }
};

struct RigidGradientContactTerms {
    Vec3 translation, orientation;
    RigidGradientContactTerms() {}
    template <RigidDerivativeMode Mode>
    void assign(const RigidEnergyDerivatives& value) {
        translation = value.translation_gradient;
        orientation = value.orientation_gradient;
    }
    template <RigidDerivativeMode Mode>
    void add(RigidEnergyDerivatives& total) const {
        total.translation_gradient += translation;
        total.orientation_gradient += orientation;
    }
};

struct RigidFullContactTerms {
    Vec3 translation, orientation;
    Mat33 tt, to, oo;
    RigidFullContactTerms() {}
    template <RigidDerivativeMode Mode>
    void assign(const RigidEnergyDerivatives& value) {
        translation = value.translation_gradient;
        orientation = value.orientation_gradient;
        tt = value.translation_translation_hessian;
        to = value.translation_orientation_hessian;
        oo = value.orientation_orientation_hessian;
    }
    template <RigidDerivativeMode Mode>
    void add(RigidEnergyDerivatives& total) const {
        total.translation_gradient += translation;
        total.orientation_gradient += orientation;
        total.translation_translation_hessian += tt;
        total.translation_orientation_hessian += to;
        total.orientation_orientation_hessian += oo;
    }
};

template <class Terms, bool Friction> struct StoredRigidContact;
template <class Terms> struct StoredRigidContact<Terms, false> {
    Terms barrier;
    StoredRigidContact() {}
};
template <class Terms> struct StoredRigidContact<Terms, true> {
    Terms barrier, friction;
    StoredRigidContact() {}
};

template <RigidDerivativeMode Mode, class Terms, bool Friction,
          class EvaluateRanges>
void ordered_rigid_contact_ranges_impl(int count, bool cooperative,
    const EvaluateRanges& evaluate, RigidEnergyDerivatives& barrier,
    RigidEnergyDerivatives* friction, int alignment,
    const std::function<void()>* leader_work) {
    using Value = StoredRigidContact<Terms, Friction>;
    ordered_sparse_contact_ranges<Value>(count, cooperative,
        [&](int begin, int end, const auto& emit) {
            evaluate(begin, end, [&](std::size_t index,
                const RigidEnergyDerivatives& b,
                const RigidEnergyDerivatives& f) {
                Value value;
                value.barrier.template assign<Mode>(b);
                if constexpr (Friction) value.friction.template assign<Mode>(f);
                emit(index, value);
            });
        }, [&](const Value& value) {
            value.barrier.template add<Mode>(barrier);
            if constexpr (Friction) value.friction.template add<Mode>(*friction);
        // Larger rigid-contact ranges amortize rejected candidates and fill
        // SIMD tiles. CCD and other users keep the default fine grain.
        }, alignment, leader_work, 1024);
}

template <RigidDerivativeMode Mode, class Terms, class EvaluateRanges>
void ordered_rigid_contact_ranges_mode(int count, bool cooperative,
    const EvaluateRanges& evaluate, RigidEnergyDerivatives& barrier,
    RigidEnergyDerivatives* friction, int alignment,
    const std::function<void()>* leader_work) {
    if (friction)
        ordered_rigid_contact_ranges_impl<Mode, Terms, true>(count, cooperative,
            evaluate, barrier, friction, alignment, leader_work);
    else
        ordered_rigid_contact_ranges_impl<Mode, Terms, false>(count, cooperative,
            evaluate, barrier, nullptr, alignment, leader_work);
}

template <class EvaluateRanges>
void ordered_rigid_contact_ranges(int count, bool cooperative,
    RigidDerivativeMode mode, const EvaluateRanges& evaluate,
    RigidEnergyDerivatives& barrier, RigidEnergyDerivatives* friction,
    int alignment = 64, const std::function<void()>* leader_work = nullptr) {
    switch (mode) {
    case RigidDerivativeMode::TranslationHessian:
        ordered_rigid_contact_ranges_mode<RigidDerivativeMode::TranslationHessian,
            RigidBlockContactTerms>(count, cooperative, evaluate, barrier,
                friction, alignment, leader_work);
        break;
    case RigidDerivativeMode::OrientationHessian:
        ordered_rigid_contact_ranges_mode<RigidDerivativeMode::OrientationHessian,
            RigidBlockContactTerms>(count, cooperative, evaluate, barrier,
                friction, alignment, leader_work);
        break;
    case RigidDerivativeMode::Gradient:
        ordered_rigid_contact_ranges_mode<RigidDerivativeMode::Gradient,
            RigidGradientContactTerms>(count, cooperative, evaluate, barrier,
                friction, alignment, leader_work);
        break;
    default:
        ordered_rigid_contact_ranges_mode<RigidDerivativeMode::Full,
            RigidFullContactTerms>(count, cooperative, evaluate, barrier,
                friction, alignment, leader_work);
        break;
    }
}

} // namespace solver_detail
