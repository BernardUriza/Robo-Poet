#!/usr/bin/env python3
"""
Input Validation Utilities for Robo-Poet Framework

Provides robust user input validation for academic interface.

Author: ML Academic Framework
Version: 2.1
"""

from typing import Union, List, Optional


class InputValidator:
    """Handles user input validation for academic interface."""
    
    @staticmethod
    def get_number_input(prompt: str, default: int, min_val: int = 1, max_val: int = 1000) -> int:
        """Get valid integer input from user with bounds checking."""
        while True:
            try:
                user_input = input(f"{prompt} [default: {default}]: ").strip()
                if user_input == "":
                    return default
                
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"❌ Valor debe estar entre {min_val} y {max_val}")
            except ValueError:
                print("❌ Por favor ingresa un número válido")
            except KeyboardInterrupt:
                print("\n❌ Operación cancelada")
                return default
    
    @staticmethod
    def get_float_input(prompt: str, default: float, min_val: float = 0.1, max_val: float = 2.0) -> float:
        """Get valid float input from user with bounds checking."""
        while True:
            try:
                user_input = input(f"{prompt} [default: {default}]: ").strip()
                if user_input == "":
                    return default
                
                value = float(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"❌ Valor debe estar entre {min_val} y {max_val}")
            except ValueError:
                print("❌ Por favor ingresa un número decimal válido")
            except KeyboardInterrupt:
                print("\n❌ Operación cancelada")
                return default
    
    @staticmethod
    def get_text_input(prompt: str, default: str = "", min_length: int = 1, 
                      max_length: int = 500) -> str:
        """Get valid text input from user with length validation."""
        while True:
            try:
                user_input = input(f"{prompt} [default: '{default}']: ").strip()
                if user_input == "" and default:
                    return default
                
                if min_length <= len(user_input) <= max_length:
                    return user_input
                else:
                    print(f"❌ Texto debe tener entre {min_length} y {max_length} caracteres")
            except KeyboardInterrupt:
                print("\n❌ Operación cancelada")
                return default
    
    @staticmethod
    def get_choice_input(prompt: str, valid_choices: List[str], 
                        default: Optional[str] = None) -> str:
        """Get valid choice from list of options."""
        choices_str = "/".join(valid_choices)
        default_text = f" [default: {default}]" if default else ""
        
        while True:
            try:
                user_input = input(f"{prompt} ({choices_str}){default_text}: ").strip().lower()
                
                if user_input == "" and default:
                    return default.lower()
                
                if user_input in [choice.lower() for choice in valid_choices]:
                    return user_input
                else:
                    print(f"❌ Opción inválida. Opciones válidas: {choices_str}")
            except KeyboardInterrupt:
                print("\n❌ Operación cancelada")
                return default.lower() if default else valid_choices[0].lower()
    
    @staticmethod
    def get_menu_choice(prompt: str, max_option: int, min_option: int = 1) -> int:
        """Get valid menu choice from user."""
        while True:
            try:
                user_input = input(f"{prompt} ({min_option}-{max_option}): ").strip()
                
                if user_input == "":
                    continue
                
                choice = int(user_input)
                if min_option <= choice <= max_option:
                    return choice
                else:
                    print(f"❌ Opción debe estar entre {min_option} y {max_option}")
            except ValueError:
                print("❌ Por favor ingresa un número válido")
            except KeyboardInterrupt:
                print("\n❌ Operación cancelada")
                return min_option
    
    @staticmethod
    def get_confirmation(prompt: str, default_yes: bool = False) -> bool:
        """Get yes/no confirmation from user."""
        default_text = " [Y/n]" if default_yes else " [y/N]"
        
        while True:
            try:
                user_input = input(f"{prompt}{default_text}: ").strip().lower()
                
                if user_input == "":
                    return default_yes
                
                if user_input in ['y', 'yes', 'sí', 'si']:
                    return True
                elif user_input in ['n', 'no']:
                    return False
                else:
                    print("❌ Responde con 'y' (sí) o 'n' (no)")
            except KeyboardInterrupt:
                print("\n❌ Operación cancelada")
                return False
    
    @staticmethod
    def get_file_path_input(prompt: str, must_exist: bool = True) -> Optional[str]:
        """Get valid file path from user."""
        from pathlib import Path
        
        while True:
            try:
                user_input = input(f"{prompt}: ").strip()
                
                if user_input == "":
                    return None
                
                path = Path(user_input)
                
                if must_exist and not path.exists():
                    print(f"❌ Archivo no encontrado: {user_input}")
                    continue
                
                return str(path)
            except KeyboardInterrupt:
                print("\n❌ Operación cancelada")
                return None
    
    @staticmethod
    def validate_epochs_input(epochs: int) -> bool:
        """Validate epochs input for academic purposes."""
        if epochs < 1:
            print("❌ Número de épocas debe ser al menos 1")
            return False
        
        if epochs > 100:
            print("⚠️ Número de épocas muy alto (>100). Esto tomará mucho tiempo.")
            return InputValidator.get_confirmation("¿Continuar de todas formas?", False)
        
        if epochs > 50:
            print("⚠️ Entrenamientos largos (>50 épocas) pueden tomar varias horas.")
            return InputValidator.get_confirmation("¿Continuar?", True)
        
        return True
    
    @staticmethod
    def validate_temperature_input(temperature: float) -> bool:
        """Validate temperature input for generation."""
        if temperature <= 0:
            print("❌ Temperature debe ser mayor que 0")
            return False
        
        if temperature > 2.0:
            print("⚠️ Temperature muy alta (>2.0) puede generar texto incoherente.")
            return InputValidator.get_confirmation("¿Continuar de todas formas?", False)
        
        if temperature < 0.3:
            print("⚠️ Temperature muy baja (<0.3) generará texto muy repetitivo.")
            return InputValidator.get_confirmation("¿Continuar de todas formas?", True)
        
        return True
    
    @staticmethod
    def validate_length_input(length: int) -> bool:
        """Validate generation length input."""
        if length < 10:
            print("❌ Longitud debe ser al menos 10 caracteres")
            return False
        
        if length > 2000:
            print("⚠️ Longitud muy alta (>2000) puede tomar mucho tiempo.")
            return InputValidator.get_confirmation("¿Continuar?", True)
        
        return True