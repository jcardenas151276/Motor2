# Diccionario con los usuarios y sus contraseñas

usuarios = {"julian.velasquez@premexcorp.com" : "motor_beta12.",
"jgarcia@equitel.com.co" : "motor_beta12.",
"sandra.galan@contegral.co" : "motor_beta12.",
"atehortua3000@hotmail.com" : "motor_beta12.",
"deisycalle30@gmail.com" : "motor_beta12.",
"mateo0583@hotmail.com" : "motor_beta12.",
"calfonsoh@gmail.com" : "motor_beta12.",
"mateogomez@simplesolutions.com.co" : "motor_beta12."}

def validar_usuario(usuario, contrasena):
    """Valida si el usuario y la contraseña son correctos."""
    return usuario in usuarios and usuarios[usuario] == contrasena