// SPDX-License-Identifier: MIT

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp32/rom/uart.h"
#include "driver/gpio.h"
#include "driver/ledc.h"


#define GPIO_LED 21


static int read_num() {
	int num = 0;
	STATUS s;
	uint8_t c;
	while (1) {
		s = uart_rx_one_char(&c);
		if (s == OK) {
			//uart_tx_one_char(c);
			if (c == '\n' || c == '\r') {
				return num;
			}
			else if (c >= '0' && c <= '9') {
				num = num * 10 + ((int)(c - '0'));
			}
		}
	}

	return -1;
}


void app_main(void) {
	STATUS s;
	uint8_t c;
	int frequency = 100; // 100 Hz
	int resolution = 12;
	int value = (1 << (resolution - 1)); // 50%;

	gpio_pad_select_gpio(GPIO_LED);
	gpio_set_direction(GPIO_LED, GPIO_MODE_OUTPUT);
	gpio_set_level(GPIO_LED, 1);

	ledc_timer_config_t ledc_timer = {
		.duty_resolution = LEDC_TIMER_1_BIT + (resolution - 1),
		.freq_hz = frequency,
		.speed_mode = LEDC_HIGH_SPEED_MODE,
		.timer_num = LEDC_TIMER_0,
		.clk_cfg = LEDC_AUTO_CLK,
	};
	ledc_timer_config(&ledc_timer);

	ledc_channel_config_t ledc_channel = {
		.channel = LEDC_CHANNEL_0,
		.duty = 0,
		.gpio_num = GPIO_LED,
		.speed_mode = LEDC_HIGH_SPEED_MODE,
		.hpoint = 0,
		.timer_sel = LEDC_TIMER_0
	};
	ledc_channel_config(&ledc_channel);

	while (1) {
		s = uart_rx_one_char(&c);
		if (s == OK) {
			//uart_tx_one_char(c);
			switch (c) {
				case 'f':
					frequency = read_num();
					ledc_timer.freq_hz = frequency;
					break;
				case 'v':
					value = read_num();
					ledc_channel.duty = value;
					break;
				case 'r':
					resolution = read_num();
					if (resolution > 0 && resolution <= 20) {
						ledc_timer.duty_resolution = LEDC_TIMER_1_BIT + (resolution - 1);
						ledc_channel.duty = value;
					}
					break;
				case 'a':
					ledc_timer_config(&ledc_timer);
					ledc_channel_config(&ledc_channel);
					break;
				default:
					break;
			}
		}
	}

	vTaskDelay(portMAX_DELAY);
	vTaskDelete(NULL);
}
