FROM quay.io/keycloak/keycloak:20.0.3

# Adding sourced-theme login files
RUN mkdir /opt/keycloak/themes/sourced-theme
COPY ./apps/keycloak-sourced-theme/sourced-theme/. /opt/keycloak/themes/sourced-theme