function showPassword() {
  let password = document.getElementById("id_password1");
  let confirmpassword = document.getElementById("id_password2");
  let title = document.getElementById("pass");
  if (password.type === "password" && confirmpassword.type === "password") {
    password.type = "text";
    confirmpassword.type = "text";
    title.innerHTML =
      '<ion-icon name="eye-off"></ion-icon><span class="s-pass">Hide Password</span>';
  } else {
    password.type = "password";
    confirmpassword.type = "password";
    title.innerHTML = "Show Password";
    title.innerHTML =
      '<ion-icon name="eye"></ion-icon><span class="s-pass">Show Password</span>';
  }
}

function generatePassword() {
  var length = 12,
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@$%#",
    retVal = "";
  for (var i = 0, n = charset.length; i < length; ++i) {
    retVal += charset.charAt(Math.floor(Math.random() * n));
  }
  return retVal;
}
 