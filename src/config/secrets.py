"""
Secrets management for sensitive configuration.

Provides secure handling of secrets and credentials.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from cryptography.fernet import Fernet


logger = logging.getLogger(__name__)


class SecretsManager:
    """Manager for application secrets and credentials."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)
        self._secrets_cache: Dict[str, Any] = {}
    
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create a new one."""
        key_file = Path(".secrets_key")
        
        if key_file.exists():
            return key_file.read_bytes()
        else:
            # Generate new key
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            key_file.chmod(0o600)  # Restrict permissions
            logger.warning("Generated new encryption key for secrets")
            return key
    
    def encrypt_secret(self, value: str) -> str:
        """Encrypt a secret value."""
        encrypted = self.cipher.encrypt(value.encode())
        return encrypted.decode()
    
    def decrypt_secret(self, encrypted_value: str) -> str:
        """Decrypt a secret value."""
        decrypted = self.cipher.decrypt(encrypted_value.encode())
        return decrypted.decode()
    
    def store_secret(self, name: str, value: str, encrypted: bool = True) -> None:
        """Store a secret value."""
        if encrypted:
            stored_value = self.encrypt_secret(value)
        else:
            stored_value = value
        
        self._secrets_cache[name] = {
            'value': stored_value,
            'encrypted': encrypted
        }
    
    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value."""
        # Check cache first
        if name in self._secrets_cache:
            secret_data = self._secrets_cache[name]
            value = secret_data['value']
            if secret_data['encrypted']:
                return self.decrypt_secret(value)
            return value
        
        # Check environment variables
        env_value = os.getenv(name)
        if env_value:
            return env_value
        
        # Check secrets file
        secrets_file_value = self._get_from_secrets_file(name)
        if secrets_file_value:
            return secrets_file_value
        
        return default
    
    def _get_from_secrets_file(self, name: str) -> Optional[str]:
        """Get secret from secrets file."""
        secrets_file = Path(".secrets.json")
        
        if not secrets_file.exists():
            return None
        
        try:
            with open(secrets_file, 'r') as f:
                secrets_data = json.load(f)
            
            secret_info = secrets_data.get(name)
            if not secret_info:
                return None
            
            value = secret_info['value']
            encrypted = secret_info.get('encrypted', False)
            
            if encrypted:
                return self.decrypt_secret(value)
            return value
            
        except Exception as e:
            logger.error(f"Failed to read secrets file: {e}")
            return None
    
    def save_secrets_to_file(self, filepath: str = ".secrets.json") -> None:
        """Save cached secrets to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self._secrets_cache, f, indent=2)
            
            # Restrict file permissions
            Path(filepath).chmod(0o600)
            logger.info(f"Secrets saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
    
    def load_secrets_from_file(self, filepath: str = ".secrets.json") -> None:
        """Load secrets from file."""
        try:
            with open(filepath, 'r') as f:
                self._secrets_cache = json.load(f)
            logger.info(f"Secrets loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
    
    def delete_secret(self, name: str) -> None:
        """Delete a secret."""
        if name in self._secrets_cache:
            del self._secrets_cache[name]
    
    def list_secrets(self) -> list:
        """List all secret names (not values)."""
        return list(self._secrets_cache.keys())
    
    def clear_cache(self) -> None:
        """Clear secrets cache."""
        self._secrets_cache.clear()


class EnvironmentSecretsProvider:
    """Provider for secrets from environment variables."""
    
    @staticmethod
    def get_database_password() -> Optional[str]:
        """Get database password from environment."""
        return os.getenv('DB_PASSWORD')
    
    @staticmethod
    def get_api_secret_key() -> Optional[str]:
        """Get API secret key from environment."""
        return os.getenv('SECRET_KEY')
    
    @staticmethod
    def get_encryption_key() -> Optional[str]:
        """Get encryption key from environment."""
        return os.getenv('ENCRYPTION_KEY')
    
    @staticmethod
    def get_external_api_key(service: str) -> Optional[str]:
        """Get external API key for a service."""
        return os.getenv(f'{service.upper()}_API_KEY')


class VaultSecretsProvider:
    """Provider for secrets from HashiCorp Vault (future implementation)."""
    
    def __init__(self, vault_url: str, vault_token: str):
        self.vault_url = vault_url
        self.vault_token = vault_token
    
    def get_secret(self, path: str, key: str) -> Optional[str]:
        """Get secret from Vault."""
        # TODO: Implement Vault integration
        logger.warning("Vault integration not implemented yet")
        return None


class SecretsFactory:
    """Factory for creating secrets providers."""
    
    @staticmethod
    def create_secrets_manager(provider: str = "local") -> SecretsManager:
        """Create appropriate secrets manager."""
        if provider == "local":
            return SecretsManager()
        elif provider == "env":
            # Use environment-only secrets
            manager = SecretsManager()
            # Pre-populate with environment variables
            for key, value in os.environ.items():
                if any(secret_key in key.upper() for secret_key in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']):
                    manager.store_secret(key, value, encrypted=False)
            return manager
        else:
            raise ValueError(f"Unknown secrets provider: {provider}")


# Global secrets manager
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsFactory.create_secrets_manager()
    return _secrets_manager


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    return get_secrets_manager().get_secret(name, default)


def set_secret(name: str, value: str, encrypted: bool = True) -> None:
    """Convenience function to set a secret."""
    get_secrets_manager().store_secret(name, value, encrypted)


# Predefined secret getters
def get_database_url_with_password() -> str:
    """Get complete database URL with password."""
    base_url = os.getenv('DB_URL', 'sqlite:///./robo_poet.db')
    password = get_secret('DB_PASSWORD')
    
    if password and 'postgresql://' in base_url:
        # Insert password into PostgreSQL URL
        if '@' not in base_url:
            # URL format: postgresql://user:@host/db
            base_url = base_url.replace('://', f'://user:{password}@')
        else:
            # URL already has user, just add password
            user_host = base_url.split('@')
            if ':' not in user_host[0].split('//')[-1]:
                base_url = base_url.replace('@', f':{password}@')
    
    return base_url


def get_secure_secret_key() -> str:
    """Get secure secret key for the application with enhanced validation."""
    import re
    import secrets
    
    secret_key = get_secret('SECRET_KEY')
    if not secret_key:
        secret_key = os.getenv('SECURITY_SECRET_KEY', 'dev-secret-key-change-in-production')
    
    environment = os.getenv('ENVIRONMENT', 'development').lower()
    
    # Enhanced validation for all environments
    if secret_key == 'dev-secret-key-change-in-production':
        if environment == 'production':
            raise ValueError(
                "CRITICAL SECURITY ERROR: Default secret key detected in production. "
                "Set a secure SECRET_KEY environment variable immediately."
            )
        elif environment in ['staging', 'testing']:
            raise ValueError(
                f"Default secret key not allowed in {environment} environment. "
                "Set a secure SECRET_KEY environment variable."
            )
    
    # Validate key strength for non-development environments
    if environment != 'development':
        if len(secret_key) < 32:
            raise ValueError(f"SECRET_KEY must be at least 32 characters in {environment} environment")
        
        # Check for complexity (should have mix of characters)
        if not re.search(r'[A-Z]', secret_key):
            raise ValueError("SECRET_KEY should contain uppercase letters")
        if not re.search(r'[a-z]', secret_key):
            raise ValueError("SECRET_KEY should contain lowercase letters") 
        if not re.search(r'[0-9]', secret_key):
            raise ValueError("SECRET_KEY should contain digits")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', secret_key):
            raise ValueError("SECRET_KEY should contain special characters")
    
    # Additional security: warn if key appears to be weak
    if len(set(secret_key)) < len(secret_key) * 0.5:  # Less than 50% unique characters
        import logging
        logging.getLogger(__name__).warning(
            "SECRET_KEY has low character diversity. Consider generating a new key."
        )
    
    return secret_key