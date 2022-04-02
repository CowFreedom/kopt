#pragma once
#include <cstdint>

namespace kf::rand{
	
	constexpr std::uint64_t PCG32_MULT=0x5851f42d4c957f2dULL;

	/*Implementations of the PCG random number generator.
	See pg. 43 of "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation" by Melissa E. O'Neill
	and see also Physically Based Rendering 3 by Matt Pharr*/
	__host__ __device__
	std::uint32_t pcg_xsh_rr(std::uint64_t& state, std::uint64_t& inc){
		std::uint64_t oldstate=state;
		state=oldstate*PCG32_MULT+inc;
		//printf("O: %d\n",inc);
		std::uint32_t xorshifted=(std::uint32_t)((oldstate^(oldstate>>18u))>>27u);
		std::uint32_t rot=(std::uint32_t)(oldstate>>59u);
		return (xorshifted>>rot) | (xorshifted<<((~rot+1u)&31u));
	}

	__host__ __device__
	void set_sequence(std::uint64_t sequence_index, std::uint64_t offset, std::uint64_t& state, std::uint64_t& inc){
		inc=(sequence_index<<1u)|1u;
		state=(inc+offset)*PCG32_MULT+inc;
	}

	__host__ __device__
	float pcg_xsh_rr_real(std::uint64_t& state, std::uint64_t& inc){
		return pcg_xsh_rr(state,inc)* 0x1p-32f;
	}

	//https://mumble.net/~campbell/tmp/random_real.c
	__host__ __device__
	float pcg_xsh_rr_real(ulonglong2& pcg_data){
		return pcg_xsh_rr(pcg_data.x,pcg_data.y)* 0x1p-32f;
	}

//TODO: VIelleicht long?
}