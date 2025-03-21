import math
import time
import json
import hashlib
import secrets
import os
from typing import List, Dict, Tuple, Any, Optional
import numpy as np  # used as default; will be replaced by CuPy if available
from dataclasses import dataclass

# --------------------------------------------------
# GPU Detection and Setting xp to either CuPy or NumPy
# --------------------------------------------------
try:
    import cupy as cp
    from pycuda import driver as cuda
    import pycuda.autoinit
    GPU_AVAILABLE = True
    xp = cp  # Use CuPy for GPU arrays
    print("[INFO] GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    xp = np  # Fallback to NumPy for CPU arrays
    print("[INFO] Using CPU-only mode")

# --------------------------------------------------
# Import BN128 Parameters and Group Operations from py_ecc.bn128
# --------------------------------------------------
from py_ecc.bn128.bn128_curve import (
    field_modulus,     # Prime modulus of the base field
    b as curve_b,      # Curve coefficient (FQ(3))
    G1,                # Generator for G1 (affine coordinates)
    G2,                # Generator for G2
    multiply as ec_multiply,  # Scalar multiplication
    curve_order        # Order of the subgroup (prime order)
)
from py_ecc.bn128 import bn128_pairing
from py_ecc.bn128.bn128_curve import add, neg  # Group addition and negation

# --------------------------------------------------
# BN128 Wrapper Class
# --------------------------------------------------
class BN128:
    p = field_modulus
    b = curve_b
    G1_gen: Tuple[Any, Any] = G1
    G2_gen: Tuple[Any, Any] = G2

# --------------------------------------------------
# Global Constants and Secure Random Generation
# --------------------------------------------------
CONSTANTS = {
    'S_VAL': int.from_bytes(hashlib.sha256(b'S_VAL_SEED').digest()[:4], 'big') % curve_order,
    'H_VAL': int.from_bytes(hashlib.sha256(b'H_VAL_SEED').digest()[:4], 'big') % curve_order,
    'F_POLY_VAL': int.from_bytes(hashlib.sha256(b'F_POLY_VAL_SEED').digest()[:4], 'big') % curve_order
}

def secure_random(bits: int = 256) -> int:
    """Return a secure random integer in Z_curve_order."""
    try:
        return int.from_bytes(secrets.token_bytes(bits // 8 + (1 if bits % 8 else 0)), 'big') % curve_order
    except Exception as e:
        raise RuntimeError("Secure random generation failed") from e

# --------------------------------------------------
# Enhanced JSON Encoder
# --------------------------------------------------
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, xp.ndarray):
                return obj.get() if GPU_AVAILABLE else obj.tolist()
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (complex, np.complex128)):
                return {"real": float(obj.real), "imag": float(obj.imag)}
            return super().default(obj)
        except Exception:
            return f"[ENCODING_ERROR: {type(obj).__name__}]"

# --------------------------------------------------
# Elliptic Curve Operations for G1 and G2
# --------------------------------------------------
def ec_mul_g1(point: Tuple[Any, Any], scalar: int) -> Tuple[Any, Any]:
    if point is None:
        raise ValueError("Invalid G1 point: None")
    try:
        return ec_multiply(point, scalar)
    except Exception as e:
        raise RuntimeError("G1 scalar multiplication failed") from e

def ec_mul_g2(point: Tuple[Any, Any], scalar: int) -> Tuple[Any, Any]:
    if point is None:
        raise ValueError("Invalid G2 point: None")
    try:
        return ec_multiply(point, scalar)
    except Exception as e:
        raise RuntimeError("G2 scalar multiplication failed") from e

def pairing_full(g1: Tuple[Any, Any], g2: Tuple[Any, Any]) -> int:
    try:
        pairing_value = bn128_pairing.pairing(g2, g1)
        pairing_str = str(pairing_value)
        return int(hashlib.sha256(pairing_str.encode("utf-8")).hexdigest(), 16) % BN128.p
    except Exception as e:
        raise RuntimeError("Pairing computation failed") from e

# --------------------------------------------------
# Polynomial Helper Functions
# --------------------------------------------------
def poly_eval(coeffs: List[int], x: int) -> int:
    """Evaluate f(x)=a0+a1*x+...+ad*x^d mod curve_order."""
    result = 0
    power = 1
    for coeff in coeffs:
        result = (result + coeff * power) % curve_order
        power = (power * x) % curve_order
    return result

def poly_division(coeffs: List[int], c: int) -> List[int]:
    """
    Synthetic division of f(x)-f(c) by (x-c).
    Coeffs are in ascending order. Returns quotient coefficients.
    """
    d = len(coeffs) - 1
    rev = coeffs[::-1]
    q = [rev[0]]
    for i in range(1, d + 1):
        q.append((rev[i] + c * q[i - 1]) % curve_order)
    return q[:-1][::-1]  # Discard remainder and reverse back

# --------------------------------------------------
# Generalized KZG Polynomial Commitment (Arbitrary Degree)
# --------------------------------------------------
class KZGPoly:
    @staticmethod
    def commit(degree: int, t: int) -> Tuple[Tuple[Any, Any], List[int]]:
        """
        Generate random polynomial f(x)=a0+...+ad*x^d (degree d) and commit:
          C = g^(f(t))
        Returns the commitment and the polynomial coefficients.
        """
        try:
            coeffs = [secure_random(128) for _ in range(degree + 1)]
            f_t = poly_eval(coeffs, t)
            commitment = ec_mul_g1(BN128.G1_gen, f_t)
            return commitment, coeffs
        except Exception as e:
            raise RuntimeError("KZG polynomial commitment failed") from e

    @staticmethod
    def open(coeffs: List[int], alpha: int, t: int) -> Tuple[int, Tuple[Any, Any]]:
        """
        Open the commitment at challenge alpha.
        Compute f(alpha) and produce witness π = g^(q(t)), where
          q(x) = (f(x)-f(alpha))/(x-α)
        """
        try:
            evaluation = poly_eval(coeffs, alpha)
            q_coeffs = poly_division(coeffs, alpha)
            q_t = poly_eval(q_coeffs, t)
            proof = ec_mul_g1(BN128.G1_gen, q_t)
            return evaluation, proof
        except Exception as e:
            raise RuntimeError("KZG polynomial opening failed") from e

    @staticmethod
    def verify(commitment: Tuple[Any, Any], evaluation: int,
               proof: Tuple[Any, Any], alpha: int, t: int) -> bool:
        """
        Verify the opening using the pairing check:
          e(C / g^(f(α)), h) ?= e(π, h^(t-α))
        """
        try:
            g_falpha = ec_mul_g1(BN128.G1_gen, evaluation)
            C_div = add(commitment, neg(g_falpha))  # C - g^(f(α))
            exponent = (t - alpha) % curve_order
            h_factor = ec_mul_g2(BN128.G2_gen, exponent)
            left = pairing_full(C_div, BN128.G2_gen)
            right = pairing_full(proof, h_factor)
            result = 0
            for i in range(256):
                result |= ((left >> i) & 1) ^ ((right >> i) & 1)
            return result == 0
        except Exception as e:
            print("Error in KZG polynomial verification:", e)
            return False

# --------------------------------------------------
# Complex Transformations and Binding using xp (CuPy or NumPy)
# --------------------------------------------------
class ComplexTransformations:
    @staticmethod
    def teichmuller_lift(val: int, n: int) -> complex:
        try:
            angle = 2.0 * math.pi * ((val % n) / float(n))
            return complex(math.cos(angle), math.sin(angle))
        except Exception as e:
            raise RuntimeError("Teichmuller lift failed") from e

    @staticmethod
    def apply_flow(arr: xp.ndarray, flow_time: float) -> xp.ndarray:
        try:
            return arr * xp.exp(complex(0, flow_time))
        except Exception as e:
            raise RuntimeError("Flow application failed") from e

    @staticmethod
    def apply_flow_inverse(arr: xp.ndarray, flow_time: float) -> xp.ndarray:
        try:
            return arr * xp.exp(complex(0, -flow_time))
        except Exception as e:
            raise RuntimeError("Inverse flow application failed") from e

    @staticmethod
    def derive_binding(metadata: Dict) -> xp.ndarray:
        try:
            serialized = json.dumps(metadata, sort_keys=True).encode('utf-8')
            hash_value = hashlib.sha256(serialized).digest()
            hmac_key = b"binding_derivation_key"
            hmac_result = hmac_sha256(hmac_key, hash_value)
            binding = xp.zeros(len(hmac_result) * 4, dtype=xp.complex128)
            for i, byte in enumerate(hmac_result):
                for j in range(4):
                    bits = (byte >> (j * 2)) & 0x03
                    binding[i * 4 + j] = xp.exp(complex(0, bits * (math.pi / 2)))
            return binding
        except Exception as e:
            raise RuntimeError("Binding derivation failed") from e

    @staticmethod
    def resize_binding(binding: xp.ndarray, target_size: int) -> xp.ndarray:
        try:
            if binding.size == target_size:
                return binding
            if binding.size < 2:
                return xp.full(target_size, binding[0] if binding.size > 0 else complex(1, 0))
            indices = xp.linspace(0, binding.size - 1, target_size)
            binding_resized = xp.zeros(target_size, dtype=xp.complex128)
            idx_floor = xp.floor(indices).astype(int)
            idx_ceil = xp.minimum(xp.ceil(indices).astype(int), binding.size - 1)
            t = indices - idx_floor
            same_idx = (idx_floor == idx_ceil)
            binding_resized[same_idx] = binding[idx_floor[same_idx]]
            interp_idx = ~same_idx
            binding_resized[interp_idx] = binding[idx_floor[interp_idx]] * (1 - t[interp_idx]) + binding[idx_ceil[interp_idx]] * t[interp_idx]
            return binding_resized
        except Exception as e:
            raise RuntimeError("Binding resize failed") from e

    @staticmethod
    def apply_binding(base_arr: xp.ndarray, metadata: Dict, strength: float = 0.3) -> xp.ndarray:
        try:
            binding = ComplexTransformations.derive_binding(metadata)
            if binding.size != base_arr.size:
                binding = ComplexTransformations.resize_binding(binding, base_arr.size)
            norm = xp.linalg.norm(binding)
            binding_norm = binding / norm if norm > 1e-12 else binding
            noise_factor = 1e-10
            noise = xp.random.normal(0, noise_factor, base_arr.shape) + 1j * xp.random.normal(0, noise_factor, base_arr.shape)
            return base_arr + strength * binding_norm + noise
        except Exception as e:
            raise RuntimeError("Applying binding failed") from e

    @staticmethod
    def verify_binding(bound_arr: xp.ndarray, metadata: Dict, strength: float = 0.3, threshold: float = 0.15) -> Tuple[bool, float]:
        try:
            n = metadata["n"]
            base_val = ComplexTransformations.teichmuller_lift(17, n)
            base_arr = xp.full(n, base_val, dtype=xp.complex128)
            extracted = (bound_arr - base_arr) / strength
            expected = ComplexTransformations.derive_binding(metadata)
            if expected.size != extracted.size:
                expected = ComplexTransformations.resize_binding(expected, extracted.size)
            norm_expected = xp.linalg.norm(expected)
            norm_extracted = xp.linalg.norm(extracted)
            if norm_expected > 1e-12:
                expected /= norm_expected
            if norm_extracted > 1e-12:
                extracted /= norm_extracted
            similarity = float(xp.abs(xp.vdot(extracted, expected)))
            return similarity >= threshold, similarity
        except Exception as e:
            print("Error in binding verification:", e)
            return False, 0.0

# --------------------------------------------------
# 10. HMAC-SHA256 Utility
# --------------------------------------------------
def hmac_sha256(key: bytes, message: bytes) -> bytes:
    try:
        block_size = 64
        if len(key) > block_size:
            key = hashlib.sha256(key).digest()
        key = key.ljust(block_size, b'\x00')
        o_key_pad = bytes(x ^ 0x5c for x in key)
        i_key_pad = bytes(x ^ 0x36 for x in key)
        inner = hashlib.sha256(i_key_pad + message).digest()
        return hashlib.sha256(o_key_pad + inner).digest()
    except Exception as e:
        raise RuntimeError("HMAC-SHA256 computation failed") from e

# --------------------------------------------------
# 11. Audit Logging Classes
# --------------------------------------------------
@dataclass
class AuditEntry:
    timestamp: float
    session_id: str
    op_type: str
    payload: Any
    prev_hash: str = ""

    def __post_init__(self):
        try:
            payload_repr = repr(self.payload).encode("utf-8", "ignore")
            self.payload_hash = hashlib.sha256(payload_repr).hexdigest()
            self.payload_preview = str(self.payload)[:128]
            chain_data = f"{self.timestamp}|{self.session_id}|{self.op_type}|{self.payload_hash}|{self.prev_hash}".encode("utf-8")
            self.chain_hash = hashlib.sha256(chain_data).hexdigest()
        except Exception as e:
            raise RuntimeError("AuditEntry initialization failed") from e

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "op_type": self.op_type,
            "payload_hash": self.payload_hash,
            "payload_preview": self.payload_preview,
            "chain_hash": self.chain_hash,
            "prev_hash": self.prev_hash,
            "full_payload": self.payload
        }
    
    def validate_hash(self) -> bool:
        chain_data = f"{self.timestamp}|{self.session_id}|{self.op_type}|{self.payload_hash}|{self.prev_hash}".encode("utf-8")
        return hashlib.sha256(chain_data).hexdigest() == self.chain_hash

class AuditLog:
    def __init__(self):
        self.entries: List[AuditEntry] = []
        self.last_hash: str = hashlib.sha256(b"GENESIS").hexdigest()

    def record(self, op_type: str, session_id: str, payload: Any) -> AuditEntry:
        try:
            entry = AuditEntry(time.time(), session_id, op_type, payload, self.last_hash)
            self.entries.append(entry)
            self.last_hash = entry.chain_hash
            return entry
        except Exception as e:
            raise RuntimeError("Failed to record audit log entry") from e

    def show_log(self) -> None:
        for i, entry in enumerate(self.entries):
            print(f"[{i}] T={entry.timestamp:.4f}, Sess={entry.session_id}, Op={entry.op_type}, "
                  f"Hash={entry.payload_hash[:16]}..., Preview={entry.payload_preview}")

    def dump_json(self, filename: str) -> None:
        try:
            with open(filename, "w") as f:
                json.dump([entry.to_dict() for entry in self.entries], f, indent=2, cls=EnhancedJSONEncoder)
        except Exception as e:
            raise RuntimeError("Failed to dump audit log to file") from e

    def verify_integrity(self) -> Tuple[bool, Optional[int]]:
        try:
            if not self.entries:
                return True, None
            expected = hashlib.sha256(b"GENESIS").hexdigest()
            for i, entry in enumerate(self.entries):
                if entry.prev_hash != expected or not entry.validate_hash():
                    return False, i
                expected = entry.chain_hash
            return True, None
        except Exception as e:
            raise RuntimeError("Audit log integrity verification failed") from e

# --------------------------------------------------
# 12. Prover Session and MultiProver Aggregator Classes
# --------------------------------------------------
class ProverSession:
    def __init__(self, aggregator: "MultiProverAggregator", session_id: str, n: int = 128, chunk_size: int = 64, poly_degree: int = 3):
        self.agg = aggregator
        self.log = aggregator.log
        self.session_id = session_id
        self.n = n
        self.chunk_size = chunk_size
        self.poly_degree = poly_degree
        self.use_gpu = GPU_AVAILABLE
        self.session_key = secrets.token_hex(32)
        self.creation_time = time.time()
        self.last_activity = self.creation_time
        self.activity_count = 0
        self.status = "INITIALIZED"
        # Trusted setup parameter t
        self.t = secure_random(128)
        print(f"[INFO] {'GPU' if self.use_gpu else 'CPU'} mode for session {session_id} with t = {self.t}")

    def _update_activity(self):
        self.last_activity = time.time()
        self.activity_count += 1

    def commit(self) -> Tuple[Tuple[Any, Any], xp.ndarray, float, Dict]:
        try:
            self._update_activity()
            commit_val, coeffs = KZGPoly.commit(self.poly_degree, self.t)
            base_arr = xp.full(self.n, ComplexTransformations.teichmuller_lift(17, self.n), dtype=xp.complex128)
            metadata = {
                "session_id": self.session_id,
                "commit_timestamp": time.time(),
                "n": self.n,
                "chunk_size": self.chunk_size,
                "commitment_value": str(commit_val),
                "metadata_version": "2.0",
                "secure_nonce": secrets.token_hex(16),
                "poly_coeffs": coeffs,
                "poly_degree": self.poly_degree,
                "t": self.t
            }
            bound_arr = ComplexTransformations.apply_binding(base_arr, metadata)
            seed_hash = hashlib.sha256(f"{self.session_id}:{self.session_key}:{self.creation_time}".encode('utf-8')).digest()
            flow_t = float(int.from_bytes(seed_hash[:4], 'big')) / 9999999.0
            final_arr = ComplexTransformations.apply_flow(bound_arr, flow_t)
            self.log.record("COMMIT", self.session_id, {
                "commit_val": str(commit_val),
                "n": self.n,
                "flow_time": flow_t,
                "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest(),
                "activity_count": self.activity_count
            })
            self.status = "COMMITTED"
            return commit_val, final_arr, flow_t, metadata
        except Exception as e:
            raise RuntimeError("Commit phase in ProverSession failed") from e

    def respond(self, alpha: int, final_arr: xp.ndarray, flow_t: float, metadata: Dict) -> Tuple[int, Tuple[Any, Any]]:
        try:
            self._update_activity()
            if self.status != "COMMITTED":
                raise ValueError(f"Invalid session state: {self.status}. Expected: COMMITTED")
            evaluation, proof = KZGPoly.open(metadata["poly_coeffs"], alpha, self.t)
            challenge_binding = hmac_sha256(
                f"{alpha}:{self.session_id}".encode('utf-8'),
                f"{self.session_key}".encode('utf-8')
            ).hex()
            self.log.record("RESPONSE", self.session_id, {
                "evaluation": evaluation,
                "proof": str(proof),
                "metadata_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest(),
                "challenge_binding": challenge_binding,
                "activity_count": self.activity_count
            })
            self.status = "RESPONDED"
            return evaluation, proof
        except Exception as e:
            raise RuntimeError("Response phase in ProverSession failed") from e

class MultiProverAggregator:
    def __init__(self):
        try:
            self.log = AuditLog()
            self.sessions = {}
            self.creation_time = time.time()
            self.agg_key = secrets.token_hex(32)
            self.challenge_nonce = secrets.token_bytes(32)
            self.verification_results = {}
        except Exception as e:
            raise RuntimeError("Failed to initialize MultiProverAggregator") from e

    def new_session(self, sid: str, n: int = 128, chunk_size: int = 64, poly_degree: int = 3) -> ProverSession:
        if sid in self.sessions:
            raise ValueError(f"Session ID '{sid}' already exists")
        session = ProverSession(self, sid, n, chunk_size, poly_degree)
        self.sessions[sid] = session
        self.log.record("SESSION_CREATE", sid, {
            "n": n,
            "chunk_size": chunk_size,
            "poly_degree": poly_degree,
            "creation_time": session.creation_time
        })
        return session

    def challenge(self) -> int:
        try:
            challenge_data = f"{self.agg_key}:{time.time()}:{len(self.sessions)}".encode('utf-8')
            challenge_data += self.challenge_nonce
            hash_digest = hashlib.sha256(challenge_data).digest()
            alpha = 2 + (int.from_bytes(hash_digest[:4], 'big') % (curve_order - 3))
            self.log.record("CHALLENGE", "ALL", {
                "alpha": alpha,
                "challenge_time": time.time(),
                "active_sessions": list(self.sessions.keys())
            })
            self.challenge_nonce = hmac_sha256(self.challenge_nonce, challenge_data)
            return alpha
        except Exception as e:
            raise RuntimeError("Challenge generation failed") from e

    def verify(self, sid: str, commit_val: Tuple[Any, Any], alpha: int, 
               evaluation: int, proof: Tuple[Any, Any], final_arr: xp.ndarray, 
               flow_t: float, metadata: Dict) -> Tuple[bool, Dict]:
        if sid not in self.sessions:
            raise ValueError(f"Unknown session ID: {sid}")
        try:
            ok_kzg = KZGPoly.verify(commit_val, evaluation, proof, alpha, metadata["t"])
            undone = ComplexTransformations.apply_flow_inverse(final_arr, flow_t)
            binding_ok, similarity = ComplexTransformations.verify_binding(undone, metadata)
            mean_mag = float(xp.mean(xp.abs(undone)).get() if GPU_AVAILABLE else xp.mean(xp.abs(undone)))
            magnitude_diff = abs(mean_mag - 1.0)
            ok_flow = (magnitude_diff < 0.02)
            metadata_ok = (
                metadata.get("session_id") == sid and
                "commit_timestamp" in metadata and
                metadata.get("n") == self.sessions[sid].n and
                metadata.get("chunk_size") == self.sessions[sid].chunk_size
            )
            result = ok_kzg and ok_flow and binding_ok and metadata_ok
            details = {
                "kzg_ok": bool(ok_kzg),
                "flow_ok": bool(ok_flow),
                "binding_ok": bool(binding_ok),
                "metadata_ok": bool(metadata_ok),
                "binding_similarity": float(similarity),
                "magnitude_diff": float(magnitude_diff),
                "verification_time": time.time(),
                "result": bool(result)
            }
            self.verification_results[sid] = details
            self.log.record("VERIFY", sid, details)
            return result, details
        except Exception as e:
            raise RuntimeError("Verification in aggregator failed") from e

    def sign_final(self, priv_key: int = None) -> str:
        try:
            if priv_key is None:
                priv_key_bytes = hashlib.sha256(self.agg_key.encode('utf-8')).digest()[:16]
                priv_key = int.from_bytes(priv_key_bytes, 'big')
            log_ok, tamper_idx = self.log.verify_integrity()
            if not log_ok:
                raise Exception(f"Audit log integrity check failed at index {tamper_idx}")
            payload = "|".join(entry.chain_hash for entry in self.log.entries)
            payload += f"|{self.creation_time}|{time.time()}"
            hv = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            sig = pow(int(hv, 16), priv_key, BN128.p)
            self.log.record("SIGN_FINAL", "AGGREGATOR", {
                "final_hash": hv,
                "signature": hex(sig),
                "log_entries": len(self.log.entries),
                "verification_results": len(self.verification_results),
                "signing_time": time.time()
            })
            return hex(sig)
        except Exception as e:
            raise RuntimeError("Final signing failed") from e

    def dump_log(self, fname: str) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(fname)) if os.path.dirname(fname) else '.', exist_ok=True)
            log_ok, tamper_idx = self.log.verify_integrity()
            self.log.record("LOG_DUMP", "AGGREGATOR", {
                "filename": fname,
                "entries": len(self.log.entries),
                "integrity_check": log_ok,
                "tamper_index": tamper_idx
            })
            self.log.dump_json(fname)
        except Exception as e:
            raise RuntimeError("Failed to dump audit log") from e

class SecurityError(Exception):
    """Raised for security-related issues."""
    pass

# --------------------------------------------------
# 13. Demonstration Function
# --------------------------------------------------
def run_demonstration():
    try:
        print("=== Starting Enhanced Cryptographic Protocol Demonstration ===\n")
        aggregator = MultiProverAggregator()

        print("=== Prover-Alpha: Commitment Phase ===")
        session_a = aggregator.new_session("prover-alpha", 128, 64, poly_degree=5)
        commit_val_a, array_a, flow_a, metadata_a = session_a.commit()

        print("\n=== Prover-Beta: Commitment Phase ===")
        session_b = aggregator.new_session("prover-beta", 128, 64, poly_degree=5)
        commit_val_b, array_b, flow_b, metadata_b = session_b.commit()

        print("\n=== Aggregator: Challenge Generation ===")
        alpha = aggregator.challenge()
        print(f"Generated challenge: {alpha}")

        print("\n=== Prover-Alpha: Proof Generation ===")
        evaluation_a, proof_a = session_a.respond(alpha, array_a, flow_a, metadata_a)
        print(f"  f({alpha}) = {evaluation_a}")
        print(f"  Proof: {proof_a}")

        print("\n=== Prover-Beta: Proof Generation ===")
        evaluation_b, proof_b = session_b.respond(alpha, array_b, flow_b, metadata_b)
        print(f"  f({alpha}) = {evaluation_b}")
        print(f"  Proof: {proof_b}")

        print("\n=== Aggregator: Verification for Prover-Alpha ===")
        result_a, details_a = aggregator.verify(
            "prover-alpha", commit_val_a, alpha, evaluation_a, proof_a, array_a, flow_a, metadata_a
        )
        print(f"Verification result: {'Success' if result_a else 'Failure'}")
        print(f"Details: {json.dumps(details_a, indent=2, cls=EnhancedJSONEncoder)}")

        print("\n=== Aggregator: Verification for Prover-Beta ===")
        result_b, details_b = aggregator.verify(
            "prover-beta", commit_val_b, alpha, evaluation_b, proof_b, array_b, flow_b, metadata_b
        )
        print(f"Verification result: {'Success' if result_b else 'Failure'}")
        print(f"Details: {json.dumps(details_b, indent=2, cls=EnhancedJSONEncoder)}")

        print("\n=== Audit Log Verification ===")
        log_ok, tamper_idx = aggregator.log.verify_integrity()
        print(f"Log integrity check: {'Passed' if log_ok else f'Failed at index {tamper_idx}'}")

        print("\n=== Audit Log Entries ===")
        aggregator.log.show_log()

        sig_hex = aggregator.sign_final()
        print("\nAggregator final signature (hex) =", sig_hex)

        log_file = os.path.join("logs", f"protocol_run_{int(time.time())}.json")
        aggregator.dump_log(log_file)
        print(f"[Log saved to {log_file}]")
    except Exception as e:
        print("An error occurred during demonstration:", e)

if __name__ == "__main__":
    run_demonstration()



